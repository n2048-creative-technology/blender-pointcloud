import logging
import math
from dataclasses import dataclass

import bpy
from bpy.props import (
    BoolProperty,
    EnumProperty,
    FloatProperty,
    IntProperty,
    PointerProperty,
    StringProperty,
)
from bpy.types import Operator, Panel, PropertyGroup

try:  # RealSense SDK is optional at import time
    import pyrealsense2 as rs
except Exception:  # pragma: no cover - Blender runtime only
    rs = None

try:
    import numpy as np
except Exception:  # pragma: no cover - Blender runtime only
    np = None


logger = logging.getLogger(__name__)

POINT_DISPLAY_GROUP_NAME = "RealSense_Point_Display"
POINT_DISPLAY_VALUE_NODE = "RS_PointSize"


def runtime_state():
    return _RUNTIME


@dataclass
class _RuntimeState:
    session: "RealSenseSession | None" = None
    status_message: str = ""


_RUNTIME = _RuntimeState()


class RealSenseError(RuntimeError):
    pass


def ensure_numpy_available():
    if np is None:
        raise RealSenseError("numpy is required for point cloud conversion")


def ensure_realsense_available():
    if rs is None:
        raise RealSenseError("pyrealsense2 is not installed or failed to load")


def ensure_collection_linked(collection, scene):
    children = scene.collection.children
    if collection.name not in children.keys():
        children.link(collection)


def ensure_point_display_group(point_size):
    group = bpy.data.node_groups.get(POINT_DISPLAY_GROUP_NAME)
    if group is None:
        group = bpy.data.node_groups.new(POINT_DISPLAY_GROUP_NAME, 'GeometryNodeTree')
        interface = group.interface
        interface.new_socket("Geometry", in_out='INPUT', socket_type='NodeSocketGeometry')
        interface.new_socket("Geometry", in_out='OUTPUT', socket_type='NodeSocketGeometry')
        input_node = group.nodes.new("NodeGroupInput")
        output_node = group.nodes.new("NodeGroupOutput")
        output_node.is_active_output = True
        mesh_to_points = group.nodes.new("GeometryNodeMeshToPoints")
        mesh_to_points.mode = 'VERTICES'
        value_node = group.nodes.new("ShaderNodeValue")
        value_node.name = POINT_DISPLAY_VALUE_NODE
        links = group.links
        links.new(input_node.outputs["Geometry"], mesh_to_points.inputs["Mesh"])
        links.new(value_node.outputs[0], mesh_to_points.inputs["Radius"])
        links.new(mesh_to_points.outputs["Points"], output_node.inputs["Geometry"])
    value_node = group.nodes.get(POINT_DISPLAY_VALUE_NODE)
    if value_node:
        value_node.outputs[0].default_value = max(1e-5, float(point_size))
    return group


def ensure_point_display_modifier(obj, point_size):
    if obj is None:
        return None
    group = ensure_point_display_group(point_size)
    mod_name = "RealSense Point Display"
    modifier = obj.modifiers.get(mod_name)
    if modifier is None or modifier.type != 'NODES':
        modifier = obj.modifiers.new(mod_name, 'NODES')
    modifier.node_group = group
    modifier.show_render = False
    modifier.show_viewport = True
    return modifier


def set_option_if_supported(target, option, value):
    if target is None or option is None:
        return
    supports = False
    try:
        supports = option in target.get_supported_options()
    except AttributeError:
        try:
            supports = bool(target.supports(option))
        except AttributeError:
            supports = False
    if not supports:
        return
    try:
        target.set_option(option, float(value))
    except Exception as exc:
        logger.debug("Failed to set option %s: %s", option, exc)


def rs_option(name):
    if rs is None:
        return None
    return getattr(rs.option, name, None)


class RealSenseSession:
    def __init__(self, context, settings):
        ensure_realsense_available()
        ensure_numpy_available()
        self.context = context
        self.scene = context.scene
        self.settings = settings
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.pointcloud = rs.pointcloud()
        self.align = rs.align(rs.stream.color)
        self.decimation_filter = rs.decimation_filter()
        self.spatial_filter = rs.spatial_filter()
        self.temporal_filter = rs.temporal_filter()
        self.hole_filling_filter = rs.hole_filling_filter()
        self._timer_active = False
        self._mesh = None
        self._object = None
        self._collection = None
        self._point_modifier = None
        self.last_vertices = []
        self._configure_streams()

    @property
    def is_running(self):
        return self._timer_active

    def _configure_streams(self):
        cfg = self.config
        cfg.enable_stream(
            rs.stream.depth,
            self.settings.depth_width,
            self.settings.depth_height,
            rs.format.z16,
            self.settings.depth_fps,
        )
        cfg.enable_stream(
            rs.stream.color,
            self.settings.color_width,
            self.settings.color_height,
            rs.format.rgb8,
            self.settings.color_fps,
        )

    def start(self):
        try:
            self.profile = self.pipeline.start(self.config)
        except Exception as exc:
            raise RealSenseError(f"Could not start RealSense pipeline: {exc}") from exc
        self._apply_sensor_options()
        self._ensure_live_object()
        self._timer_active = True
        bpy.app.timers.register(self._timer_callback, first_interval=0.0)

    def stop(self):
        self._timer_active = False
        try:
            self.pipeline.stop()
        except Exception:
            pass

    def snapshot(self):
        if not self.last_vertices:
            raise RealSenseError("No point cloud data available to snapshot")
        collection_name = self.settings.snapshot_collection_name
        collection = bpy.data.collections.get(collection_name)
        if collection is None:
            collection = bpy.data.collections.new(collection_name)
        ensure_collection_linked(collection, self.scene)
        snapshot_index = len(collection.objects) + 1
        mesh = bpy.data.meshes.new(f"RS_Snapshot_{snapshot_index:03d}")
        mesh.from_pydata(self.last_vertices, [], [])
        mesh.update()
        obj = bpy.data.objects.new(mesh.name, mesh)
        collection.objects.link(obj)
        ensure_point_display_modifier(obj, self.settings.point_size)
        return obj

    def _timer_callback(self):
        if not self._timer_active:
            return None
        try:
            self._update_pointcloud()
        except Exception as exc:  # pragma: no cover - Blender runtime only
            logger.exception("RealSense update failed: %s", exc)
            runtime_state().status_message = f"Stream error: {exc}"
            self.stop()
            return None
        refresh = max(0.01, 1.0 / max(1.0, self.settings.refresh_rate))
        return refresh

    def _update_pointcloud(self):
        frames = self.pipeline.poll_for_frames()
        if not frames:
            return
        if self.settings.align_to_color:
            frames = self.align.process(frames)
        depth_frame = frames.get_depth_frame()
        if not depth_frame:
            return
        color_frame = frames.get_color_frame()
        depth_frame = self._apply_filters(depth_frame)
        points = self.pointcloud.calculate(depth_frame)
        if color_frame:
            self.pointcloud.map_to(color_frame)
        vertices = np.asarray(points.get_vertices()).view(np.float32).reshape(-1, 3)
        if not len(vertices):
            return
        mask = np.isfinite(vertices).all(axis=1)
        vertices = vertices[mask]
        if not len(vertices):
            return
        min_depth = max(0.0, float(self.settings.min_depth))
        max_depth = max(0.0, float(self.settings.max_depth))
        if max_depth > 0.0 and min_depth > 0.0 and max_depth < min_depth:
            min_depth, max_depth = max_depth, min_depth
        if min_depth > 0.0:
            vertices = vertices[vertices[:, 2] >= min_depth]
        if max_depth > 0.0:
            vertices = vertices[vertices[:, 2] <= max_depth]
        if not len(vertices):
            return
        max_points = self.settings.max_points
        if max_points > 0 and len(vertices) > max_points:
            step = max(1, math.ceil(len(vertices) / max_points))
            vertices = vertices[::step]
        verts_list = vertices.tolist()
        self.last_vertices = [tuple(v) for v in verts_list]
        self._ensure_live_object()
        self._update_point_display()
        mesh = self._mesh
        mesh.clear_geometry()
        mesh.from_pydata(self.last_vertices, [], [])
        mesh.update()

    def _apply_filters(self, depth_frame):
        if self.settings.use_decimation:
            set_option_if_supported(
                self.decimation_filter,
                rs_option("filter_magnitude"),
                float(self.settings.decimation_magnitude),
            )
            depth_frame = self.decimation_filter.process(depth_frame)
        if self.settings.use_spatial:
            spatial = self.spatial_filter
            set_option_if_supported(spatial, rs_option("filter_alpha"), float(self.settings.spatial_alpha))
            set_option_if_supported(spatial, rs_option("filter_delta"), float(self.settings.spatial_delta))
            set_option_if_supported(spatial, rs_option("holes_fill"), float(self.settings.spatial_hole_filling))
            depth_frame = spatial.process(depth_frame)
        if self.settings.use_temporal:
            temporal = self.temporal_filter
            set_option_if_supported(temporal, rs_option("filter_alpha"), float(self.settings.temporal_alpha))
            set_option_if_supported(temporal, rs_option("filter_delta"), float(self.settings.temporal_delta))
            depth_frame = temporal.process(depth_frame)
        if self.settings.use_hole_filling:
            set_option_if_supported(
                self.hole_filling_filter,
                rs_option("holes_fill"),
                float(self.settings.hole_filling_mode),
            )
            depth_frame = self.hole_filling_filter.process(depth_frame)
        return depth_frame

    def _apply_sensor_options(self):
        device = self.profile.get_device()
        depth_sensor = device.first_depth_sensor()
        color_sensor = None
        for sensor in device.sensors:
            try:
                if sensor.get_info(rs.camera_info.name) == "RGB Camera":
                    color_sensor = sensor
                    break
            except Exception:
                continue

        if depth_sensor:
            set_option_if_supported(depth_sensor, rs_option("enable_auto_exposure"), float(self.settings.depth_auto_exposure))
            if not self.settings.depth_auto_exposure:
                set_option_if_supported(depth_sensor, rs_option("exposure"), float(self.settings.depth_exposure))
            set_option_if_supported(depth_sensor, rs_option("gain"), float(self.settings.depth_gain))
            set_option_if_supported(depth_sensor, rs_option("laser_power"), float(self.settings.laser_power))
            set_option_if_supported(depth_sensor, rs_option("emitter_enabled"), float(int(self.settings.emitter_mode)))

        if color_sensor:
            set_option_if_supported(color_sensor, rs_option("enable_auto_exposure"), float(self.settings.color_auto_exposure))
            if not self.settings.color_auto_exposure:
                set_option_if_supported(color_sensor, rs_option("exposure"), float(self.settings.color_exposure))
            set_option_if_supported(color_sensor, rs_option("gain"), float(self.settings.color_gain))

    def _ensure_live_object(self):
        if self._mesh and self._object and self._collection:
            return
        mesh = bpy.data.meshes.get(self.settings.live_mesh_name)
        if mesh is None:
            mesh = bpy.data.meshes.new(self.settings.live_mesh_name)
        obj = bpy.data.objects.get(self.settings.live_object_name)
        if obj is None:
            obj = bpy.data.objects.new(self.settings.live_object_name, mesh)
        else:
            obj.data = mesh
        collection = bpy.data.collections.get(self.settings.live_collection_name)
        if collection is None:
            collection = bpy.data.collections.new(self.settings.live_collection_name)
        ensure_collection_linked(collection, self.scene)
        if obj.name not in collection.objects.keys():
            collection.objects.link(obj)
        obj.display_type = 'WIRE'
        obj.hide_render = True
        self._mesh = mesh
        self._object = obj
        self._collection = collection
        self._update_point_display()

    def _update_point_display(self):
        modifier = ensure_point_display_modifier(self._object, self.settings.point_size)
        self._point_modifier = modifier


class RealSenseSettings(PropertyGroup):
    depth_width: IntProperty(name="Depth Width", min=320, max=1280, default=640)
    depth_height: IntProperty(name="Depth Height", min=240, max=720, default=480)
    depth_fps: IntProperty(name="Depth FPS", min=6, max=90, default=30)
    color_width: IntProperty(name="Color Width", min=320, max=1920, default=640)
    color_height: IntProperty(name="Color Height", min=240, max=1080, default=480)
    color_fps: IntProperty(name="Color FPS", min=6, max=60, default=30)
    refresh_rate: FloatProperty(name="Refresh Rate", min=1.0, max=60.0, default=15.0, subtype='NONE')
    max_points: IntProperty(name="Max Points", min=1000, max=200000, default=40000)
    align_to_color: BoolProperty(name="Align To Color", default=True)
    min_depth: FloatProperty(name="Min Depth (m)", min=0.0, max=10.0, default=0.1)
    max_depth: FloatProperty(name="Max Depth (m)", min=0.0, max=20.0, default=5.0)
    point_size: FloatProperty(name="Point Size", min=0.0005, max=0.1, default=0.005, subtype='DISTANCE')

    depth_auto_exposure: BoolProperty(name="Depth Auto Exposure", default=True)
    depth_exposure: FloatProperty(name="Depth Exposure", min=50.0, max=33000.0, default=8500.0)
    depth_gain: FloatProperty(name="Depth Gain", min=0.0, max=128.0, default=16.0)
    laser_power: FloatProperty(name="Laser Power", min=0.0, max=360.0, default=180.0)
    emitter_mode: EnumProperty(
        name="Emitter",
        items=(
            ('0', "Off", "Disable emitter"),
            ('1', "On", "Enable emitter"),
            ('2', "Auto", "Automatic emitter"),
        ),
        default='1',
    )

    color_auto_exposure: BoolProperty(name="Color Auto Exposure", default=True)
    color_exposure: FloatProperty(name="Color Exposure", min=1.0, max=33000.0, default=400.0)
    color_gain: FloatProperty(name="Color Gain", min=0.0, max=128.0, default=32.0)

    use_decimation: BoolProperty(name="Decimation", default=True)
    decimation_magnitude: IntProperty(name="Magnitude", min=1, max=8, default=2)
    use_spatial: BoolProperty(name="Spatial Filter", default=True)
    spatial_alpha: FloatProperty(name="Alpha", min=0.25, max=1.0, default=0.5)
    spatial_delta: FloatProperty(name="Delta", min=1.0, max=100.0, default=20.0)
    spatial_hole_filling: IntProperty(name="Hole Fill", min=0, max=5, default=1)
    use_temporal: BoolProperty(name="Temporal Filter", default=True)
    temporal_alpha: FloatProperty(name="Alpha", min=0.0, max=1.0, default=0.4)
    temporal_delta: FloatProperty(name="Delta", min=1.0, max=100.0, default=20.0)
    use_hole_filling: BoolProperty(name="Hole Filling", default=True)
    hole_filling_mode: IntProperty(name="Mode", min=0, max=2, default=1)

    snapshot_collection_name: StringProperty(name="Snapshot Collection", default="RealSense Snapshots")
    live_collection_name: StringProperty(name="Live Collection", default="RealSense Live")
    live_object_name: StringProperty(name="Live Object", default="RealSense Live")
    live_mesh_name: StringProperty(name="Live Mesh", default="RealSense Live Mesh")


class REALSENSE_OT_start_stream(Operator):
    bl_idname = "realsense.start_stream"
    bl_label = "Start RealSense"
    bl_description = "Start streaming the RealSense D415 point cloud"

    def execute(self, context):
        runtime = runtime_state()
        if runtime.session and runtime.session.is_running:
            self.report({'INFO'}, "RealSense stream is already active")
            return {'CANCELLED'}
        settings = context.scene.realsense_settings
        try:
            session = RealSenseSession(context, settings)
            session.start()
        except RealSenseError as exc:
            runtime.status_message = str(exc)
            self.report({'ERROR'}, runtime.status_message)
            return {'CANCELLED'}
        runtime.session = session
        runtime.status_message = "Streaming"
        self.report({'INFO'}, "RealSense stream started")
        return {'FINISHED'}


class REALSENSE_OT_stop_stream(Operator):
    bl_idname = "realsense.stop_stream"
    bl_label = "Stop RealSense"
    bl_description = "Stop the RealSense stream"

    def execute(self, context):
        runtime = runtime_state()
        session = runtime.session
        if not session:
            self.report({'INFO'}, "No active stream")
            return {'CANCELLED'}
        session.stop()
        runtime.session = None
        runtime.status_message = "Stopped"
        self.report({'INFO'}, "RealSense stream stopped")
        return {'FINISHED'}


class REALSENSE_OT_snapshot(Operator):
    bl_idname = "realsense.snapshot"
    bl_label = "Snapshot Point Cloud"
    bl_description = "Copy the live RealSense point cloud into a snapshot collection"

    def execute(self, context):
        runtime = runtime_state()
        session = runtime.session
        if not session:
            self.report({'WARNING'}, "Start the stream before taking a snapshot")
            return {'CANCELLED'}
        try:
            obj = session.snapshot()
        except RealSenseError as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        self.report({'INFO'}, f"Snapshot saved to {obj.name}")
        return {'FINISHED'}


class VIEW3D_PT_realsense_stream(Panel):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "RealSense"
    bl_label = "RealSense D415"

    def draw(self, context):
        layout = self.layout
        settings = context.scene.realsense_settings
        runtime = runtime_state()

        if rs is None:
            layout.label(text="pyrealsense2 not available", icon='ERROR')
            return
        if np is None:
            layout.label(text="numpy missing", icon='ERROR')
            return

        col = layout.column(align=True)
        col.label(text="Stream Settings")
        col.prop(settings, "depth_width")
        col.prop(settings, "depth_height")
        col.prop(settings, "depth_fps")
        col.prop(settings, "color_width")
        col.prop(settings, "color_height")
        col.prop(settings, "color_fps")
        col.prop(settings, "refresh_rate")
        col.prop(settings, "max_points")
        col.prop(settings, "align_to_color")
        col.prop(settings, "min_depth")
        col.prop(settings, "max_depth")
        col.prop(settings, "point_size")

        layout.separator()
        col = layout.column(align=True)
        col.label(text="Depth Sensor")
        col.prop(settings, "depth_auto_exposure")
        if not settings.depth_auto_exposure:
            col.prop(settings, "depth_exposure")
        col.prop(settings, "depth_gain")
        col.prop(settings, "laser_power")
        col.prop(settings, "emitter_mode", text="Emitter")

        layout.separator()
        col = layout.column(align=True)
        col.label(text="Color Sensor")
        col.prop(settings, "color_auto_exposure")
        if not settings.color_auto_exposure:
            col.prop(settings, "color_exposure")
        col.prop(settings, "color_gain")

        layout.separator()
        col = layout.column(align=True)
        col.label(text="Filtering")
        col.prop(settings, "use_decimation")
        if settings.use_decimation:
            col.prop(settings, "decimation_magnitude")
        col.prop(settings, "use_spatial")
        if settings.use_spatial:
            col.prop(settings, "spatial_alpha")
            col.prop(settings, "spatial_delta")
            col.prop(settings, "spatial_hole_filling")
        col.prop(settings, "use_temporal")
        if settings.use_temporal:
            col.prop(settings, "temporal_alpha")
            col.prop(settings, "temporal_delta")
        col.prop(settings, "use_hole_filling")
        if settings.use_hole_filling:
            col.prop(settings, "hole_filling_mode")

        layout.separator()
        col = layout.column(align=True)
        col.prop(settings, "snapshot_collection_name")

        is_running = runtime.session is not None and runtime.session.is_running
        controls = layout.row(align=True)
        start_row = controls.row(align=True)
        start_row.enabled = not is_running
        start_row.operator("realsense.start_stream", text="Start", icon='PLAY')
        stop_row = controls.row(align=True)
        stop_row.enabled = is_running
        stop_row.operator("realsense.stop_stream", text="Stop", icon='PAUSE')

        layout.operator("realsense.snapshot", text="Snapshot", icon='OUTLINER_COLLECTION')

        if runtime.status_message:
            layout.label(text=runtime.status_message, icon='INFO')


def register():
    # Ensure the pointer is always bound to the latest class definition when reloading
    if hasattr(bpy.types.Scene, "realsense_settings"):
        del bpy.types.Scene.realsense_settings
    bpy.types.Scene.realsense_settings = PointerProperty(type=RealSenseSettings)


def unregister():
    runtime = runtime_state()
    if runtime.session:
        runtime.session.stop()
        runtime.session = None
    if hasattr(bpy.types.Scene, "realsense_settings"):
        del bpy.types.Scene.realsense_settings
