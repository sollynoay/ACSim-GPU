from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
import warp as wp

import math


# ============================================================
# 1) Scene loading
# ============================================================

@dataclass
class CameraData:
    width: int
    height: int
    cam_pos: np.ndarray
    cam_right: np.ndarray
    cam_up: np.ndarray
    cam_forward: np.ndarray
    tan_half_fovx: float
    tan_half_fovy: float
    shift_x: float
    shift_y: float


@dataclass
class LightData:
    position: np.ndarray
    color: np.ndarray
    energy: float


@dataclass
class SceneData:
    mesh: wp.Mesh
    tri_object_id: wp.array
    object_diffuse: wp.array
    object_reflectivity: wp.array
    camera: CameraData
    light: LightData


def load_scene_json(scene_root: Path) -> dict:
    with open(scene_root / "scene.json", "r", encoding="utf-8") as f:
        return json.load(f)


def transform_points(points: np.ndarray, matrix_world: np.ndarray) -> np.ndarray:
    points_h = np.concatenate(
        [points, np.ones((points.shape[0], 1), dtype=np.float32)],
        axis=1,
    )
    return ((matrix_world @ points_h.T).T[:, :3]).astype(np.float32)


def compute_camera_basis(matrix_world: np.ndarray):
    R = matrix_world[:3, :3]
    t = matrix_world[:3, 3]

    cam_right = R[:, 0]
    cam_up = R[:, 1]
    cam_forward = -R[:, 2]

    cam_right = cam_right / (np.linalg.norm(cam_right) + 1e-12)
    cam_up = cam_up / (np.linalg.norm(cam_up) + 1e-12)
    cam_forward = cam_forward / (np.linalg.norm(cam_forward) + 1e-12)
    return t.astype(np.float32), cam_right.astype(np.float32), cam_up.astype(np.float32), cam_forward.astype(np.float32)


def compute_camera_projection(cam: dict):
    width = int(cam["resolution_x"])
    height = int(cam["resolution_y"])
    pixel_aspect_x = float(cam.get("pixel_aspect_x", 1.0))
    pixel_aspect_y = float(cam.get("pixel_aspect_y", 1.0))
    angle_x = float(cam["angle_x_rad"])
    angle_y = float(cam["angle_y_rad"])
    sensor_fit = str(cam.get("sensor_fit", "AUTO")).upper()

    effective_aspect = (width * pixel_aspect_x) / (height * pixel_aspect_y)
    if sensor_fit == "AUTO":
        sensor_fit = "HORIZONTAL" if effective_aspect >= 1.0 else "VERTICAL"

    if sensor_fit == "HORIZONTAL":
        tan_half_fovx = np.tan(0.5 * angle_x)
        tan_half_fovy = tan_half_fovx / effective_aspect
    elif sensor_fit == "VERTICAL":
        tan_half_fovy = np.tan(0.5 * angle_y)
        tan_half_fovx = tan_half_fovy * effective_aspect
    else:
        tan_half_fovx = np.tan(0.5 * angle_x)
        tan_half_fovy = np.tan(0.5 * angle_y)

    return {
        "width": width,
        "height": height,
        "tan_half_fovx": float(tan_half_fovx),
        "tan_half_fovy": float(tan_half_fovy),
        "shift_x": float(cam.get("shift_x", 0.0)),
        "shift_y": float(cam.get("shift_y", 0.0)),
    }


def build_world_mesh_and_metadata(scene_root: Path, scene_json: dict, device):
    mesh_table = {m["mesh_key"]: m for m in scene_json["meshes"]}

    all_vertices = []
    all_indices = []
    tri_object_id = []
    object_diffuse = []
    object_reflectivity = []

    object_name_to_id = {}
    vertex_offset = 0

    for inst_idx, inst in enumerate(scene_json["instances"]):
        mesh_info = mesh_table[inst["mesh_key"]]
        data = np.load(scene_root / mesh_info["file"])
        vertices = data["vertices"].astype(np.float32)
        indices = data["indices"].astype(np.int32)

        matrix_world = np.array(inst["matrix_world"], dtype=np.float32)
        vertices_world = transform_points(vertices, matrix_world)

        all_vertices.append(vertices_world)
        all_indices.append(indices + vertex_offset)
        vertex_offset += vertices.shape[0]

        obj_id = inst_idx
        tri_object_id.append(np.full((indices.shape[0],), obj_id, dtype=np.int32))

        mat_key = inst["material_key"]
        mat = scene_json["materials"][mat_key]
        object_diffuse.append(np.array(mat["diffuse_color"], dtype=np.float32))
        object_reflectivity.append(np.float32(mat["mirror_reflectivity"]))
        object_name_to_id[inst["name"]] = obj_id

    vertices = np.concatenate(all_vertices, axis=0).astype(np.float32)
    indices = np.concatenate(all_indices, axis=0).astype(np.int32)
    tri_object_id = np.concatenate(tri_object_id, axis=0).astype(np.int32)
    object_diffuse = np.stack(object_diffuse, axis=0).astype(np.float32)
    object_reflectivity = np.array(object_reflectivity, dtype=np.float32)

    mesh = wp.Mesh(
        points=wp.array(vertices, dtype=wp.vec3, device=device),
        indices=wp.array(indices.reshape(-1), dtype=wp.int32, device=device),
    )

    return (
        mesh,
        wp.array(tri_object_id, dtype=wp.int32, device=device),
        wp.array(object_diffuse, dtype=wp.vec3, device=device),
        wp.array(object_reflectivity, dtype=wp.float32, device=device),
    )


def load_scene_data(scene_root: Path, device=None) -> SceneData:
    if device is None:
        device = wp.get_preferred_device()

    scene_json = load_scene_json(scene_root)
    active_camera = scene_json["cameras"][scene_json["active_camera"]]

    mw = np.array(active_camera["matrix_world"], dtype=np.float32)
    cam_pos, cam_right, cam_up, cam_forward = compute_camera_basis(mw)
    proj = compute_camera_projection(active_camera)

    camera = CameraData(
        width=proj["width"],
        height=proj["height"],
        cam_pos=cam_pos,
        cam_right=cam_right,
        cam_up=cam_up,
        cam_forward=cam_forward,
        tan_half_fovx=proj["tan_half_fovx"],
        tan_half_fovy=proj["tan_half_fovy"],
        shift_x=proj["shift_x"],
        shift_y=proj["shift_y"],
    )

    # v1: only support one point light
    light_name, light_json = next(iter(scene_json["lights"].items()))
    light = LightData(
        position=np.array(light_json["location"], dtype=np.float32),
        color=np.array(light_json["color"], dtype=np.float32),
        energy=float(light_json["energy"]),
    )

    mesh, tri_object_id, object_diffuse, object_reflectivity = build_world_mesh_and_metadata(
        scene_root, scene_json, device
    )

    return SceneData(
        mesh=mesh,
        tri_object_id=tri_object_id,
        object_diffuse=object_diffuse,
        object_reflectivity=object_reflectivity,
        camera=camera,
        light=light,
    )


# ============================================================
# 2) Kernels
# ============================================================

@wp.kernel
def generate_primary_rays(
    width: int,
    height: int,
    cam_pos: wp.vec3,
    cam_right: wp.vec3,
    cam_up: wp.vec3,
    cam_forward: wp.vec3,
    tan_half_fovx: float,
    tan_half_fovy: float,
    shift_x: float,
    shift_y: float,
    ray_o: wp.array(dtype=wp.vec3),
    ray_d: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    x = tid % width
    y = tid // width

    sx = 2.0 * (float(x) + 0.5) / float(width) - 1.0
    sy = 1.0 - 2.0 * (float(y) + 0.5) / float(height)
    sx = sx + 2.0 * shift_x
    sy = sy + 2.0 * shift_y

    d = wp.normalize(
        cam_forward
        + sx * tan_half_fovx * cam_right
        + sy * tan_half_fovy * cam_up
    )

    ray_o[tid] = cam_pos
    ray_d[tid] = d


@wp.kernel
def trace_first_hit(
    mesh_id: wp.uint64,
    tri_object_id: wp.array(dtype=wp.int32),
    ray_o: wp.array(dtype=wp.vec3),
    ray_d: wp.array(dtype=wp.vec3),
    hit_mask: wp.array(dtype=wp.int32),
    hit_t: wp.array(dtype=wp.float32),
    hit_p: wp.array(dtype=wp.vec3),
    hit_n: wp.array(dtype=wp.vec3),
    hit_obj: wp.array(dtype=wp.int32),
):
    tid = wp.tid()
    ro = ray_o[tid]
    rd = ray_d[tid]

    q = wp.mesh_query_ray(mesh_id, ro, rd, 1.0e6)
    if q.result:
        hit_mask[tid] = 1
        hit_t[tid] = q.t
        hit_p[tid] = ro + q.t * rd
        hit_n[tid] = wp.normalize(q.normal)
        hit_obj[tid] = tri_object_id[q.face]
    else:
        hit_mask[tid] = 0
        hit_t[tid] = 0.0
        hit_p[tid] = wp.vec3(0.0, 0.0, 0.0)
        hit_n[tid] = wp.vec3(0.0, 0.0, 0.0)
        hit_obj[tid] = -1


@wp.kernel
def shade_sonar_intensity(
    ray_d: wp.array(dtype=wp.vec3),
    hit_mask: wp.array(dtype=wp.int32),
    hit_p: wp.array(dtype=wp.vec3),
    hit_n: wp.array(dtype=wp.vec3),
    hit_obj: wp.array(dtype=wp.int32),
    object_diffuse_scalar: wp.array(dtype=wp.float32),
    light_pos: wp.vec3,
    light_power: float,
    absorption_db_per_m: float,
    source_level_db: float,
    out_c: wp.array(dtype=wp.float32),
    out_light_dist: wp.array(dtype=wp.float32),
):
    tid = wp.tid()

    if hit_mask[tid] == 0:
        out_c[tid] = 0.0
        out_light_dist[tid] = 0.0
        return

    p = hit_p[tid]
    n = wp.normalize(hit_n[tid])
    rd = ray_d[tid]

    if wp.dot(rd, n) > 0.0:
        n = -n

    L = light_pos - p
    light_dist = wp.length(L)
    out_light_dist[tid] = light_dist

    if light_dist <= 1.0e-8:
        out_c[tid] = 0.0
        return

    l = L / light_dist
    cos_theta = wp.dot(l, n)

    if cos_theta <= 0.0:
        out_c[tid] = 0.0
        return

    obj_id = hit_obj[tid]
    diffuse_intensity = object_diffuse_scalar[obj_id]

    temp_I_light = (
        4.0
        * light_dist
        * light_dist
        * wp.pow(10.0, absorption_db_per_m * light_dist / 10.0)
    )

    I_light = 0.0
    if temp_I_light > 1.0e-12:
        I_light = light_power / temp_I_light
        I_light = I_light * wp.pow(10.0, source_level_db / 10.0)

    I_diffuse = diffuse_intensity * I_light * cos_theta * cos_theta
    out_c[tid] = I_diffuse


@wp.kernel
def prepare_reflection_rays(
    ray_d: wp.array(dtype=wp.vec3),
    hit_mask: wp.array(dtype=wp.int32),
    hit_p: wp.array(dtype=wp.vec3),
    hit_n: wp.array(dtype=wp.vec3),
    hit_obj: wp.array(dtype=wp.int32),
    object_reflectivity: wp.array(dtype=wp.float32),
    next_ray_o: wp.array(dtype=wp.vec3),
    next_ray_d: wp.array(dtype=wp.vec3),
    active_mask: wp.array(dtype=wp.int32),
    out_br: wp.array(dtype=wp.float32),
):
    tid = wp.tid()

    if hit_mask[tid] == 0:
        active_mask[tid] = 0
        out_br[tid] = 0.0
        next_ray_o[tid] = wp.vec3(0.0, 0.0, 0.0)
        next_ray_d[tid] = wp.vec3(0.0, 0.0, 0.0)
        return

    obj_id = hit_obj[tid]
    refl = object_reflectivity[obj_id]
    out_br[tid] = refl

    if refl <= 0.0:
        active_mask[tid] = 0
        next_ray_o[tid] = wp.vec3(0.0, 0.0, 0.0)
        next_ray_d[tid] = wp.vec3(0.0, 0.0, 0.0)
        return

    n = wp.normalize(hit_n[tid])
    rd = ray_d[tid]
    if wp.dot(n, -rd) < 0.0:
        n = -n

    refl_dir = rd - 2.0 * wp.dot(rd, n) * n
    next_ray_o[tid] = hit_p[tid] + 1.0e-3 * n
    next_ray_d[tid] = wp.normalize(refl_dir)
    active_mask[tid] = 1


# ============================================================
# 3) Driver
# ============================================================


def render_recursive(scene: SceneData, depth: int = 2):
    width = scene.camera.width
    height = scene.camera.height
    n = width * height
    device = wp.get_preferred_device()

    ray_o = wp.zeros(n, dtype=wp.vec3, device=device)
    ray_d = wp.zeros(n, dtype=wp.vec3, device=device)

    hit_mask = wp.zeros(n, dtype=wp.int32, device=device)
    hit_t = wp.zeros(n, dtype=wp.float32, device=device)
    hit_p = wp.zeros(n, dtype=wp.vec3, device=device)
    hit_n = wp.zeros(n, dtype=wp.vec3, device=device)
    hit_obj = wp.zeros(n, dtype=wp.int32, device=device)

    out_c = wp.zeros(n, dtype=wp.float32, device=device)
    out_light_dist = wp.zeros(n, dtype=wp.float32, device=device)

    next_ray_o = wp.zeros(n, dtype=wp.vec3, device=device)
    next_ray_d = wp.zeros(n, dtype=wp.vec3, device=device)
    active_mask = wp.zeros(n, dtype=wp.int32, device=device)
    out_br = wp.zeros(n, dtype=wp.float32, device=device)

    buf_dist = np.zeros((height, width, depth + 1), dtype=np.float32)
    buf_hitloc = np.zeros((height, width, 3), dtype=np.float32)
    buf_light = np.zeros((height, width, depth + 1), dtype=np.float32)
    buf_c = np.zeros((height, width, depth + 1), dtype=np.float32)
    buf_br = np.zeros((height, width, depth + 1), dtype=np.float32)

    if hasattr(scene, "object_diffuse_scalar"):
        object_diffuse_scalar = scene.object_diffuse_scalar
    else:
        diffuse_np = scene.object_diffuse.numpy().astype(np.float32)[:, 0]
        object_diffuse_scalar = wp.array(diffuse_np, dtype=wp.float32, device=device)

    wp.launch(
        generate_primary_rays,
        dim=n,
        inputs=[
            width,
            height,
            wp.vec3(*scene.camera.cam_pos),
            wp.vec3(*scene.camera.cam_right),
            wp.vec3(*scene.camera.cam_up),
            wp.vec3(*scene.camera.cam_forward),
            scene.camera.tan_half_fovx,
            scene.camera.tan_half_fovy,
            scene.camera.shift_x,
            scene.camera.shift_y,
        ],
        outputs=[ray_o, ray_d],
        device=device,
    )

    current_ray_o = ray_o
    current_ray_d = ray_d

    for d in range(depth + 1):
        wp.launch(
            trace_first_hit,
            dim=n,
            inputs=[scene.mesh.id, scene.tri_object_id, current_ray_o, current_ray_d],
            outputs=[hit_mask, hit_t, hit_p, hit_n, hit_obj],
            device=device,
        )

        wp.launch(
            shade_sonar_intensity,
            dim=n,
            inputs=[
                current_ray_d,
                hit_mask,
                hit_p,
                hit_n,
                hit_obj,
                object_diffuse_scalar,
                wp.vec3(*scene.light.position),
                float(scene.light.energy),
                1.95,
                64.0,
            ],
            outputs=[out_c, out_light_dist],
            device=device,
        )

        hit_t_np = hit_t.numpy().reshape(height, width)
        hit_p_np = hit_p.numpy().reshape(height, width, 3)
        c_np = out_c.numpy().reshape(height, width)
        light_dist_np = out_light_dist.numpy().reshape(height, width)

        buf_dist[:, :, d] = hit_t_np

        if d == 0:
            buf_hitloc[:, :, :] = hit_p_np

        buf_light[:, :, d] = light_dist_np
        buf_c[:, :, d] = c_np

        if d < depth:
            wp.launch(
                prepare_reflection_rays,
                dim=n,
                inputs=[
                    current_ray_d,
                    hit_mask,
                    hit_p,
                    hit_n,
                    hit_obj,
                    scene.object_reflectivity,
                ],
                outputs=[next_ray_o, next_ray_d, active_mask, out_br],
                device=device,
            )

            buf_br[:, :, d] = out_br.numpy().reshape(height, width)
            current_ray_o, current_ray_d = next_ray_o, next_ray_d

    return buf_dist, buf_hitloc, buf_light, buf_c, buf_br


# ============================================================
# Warp kernels for: buf_dist / buf_light / buf_c -> sum_image
# Notes:
# - noise is intentionally removed
# - hard mask is kept
# - ac_generate is replaced by soft Gaussian splatting
# - this file only processes up to sum_image
# - assumes buf_dist, buf_light, buf_c have shape (H, W, D)
# - assumes buf_c is scalar intensity, not RGB
# ============================================================


@wp.kernel
def make_mask_from_dist(
    buf_dist: wp.array(dtype=wp.float32, ndim=3),
    eps: float,
    mask: wp.array(dtype=wp.float32, ndim=3),
):
    y, x, d = wp.tid()
    if buf_dist[y, x, d] > eps:
        mask[y, x, d] = 1.0
    else:
        mask[y, x, d] = 0.0


@wp.kernel
def reverse_cumsum_masked(
    buf_dist: wp.array(dtype=wp.float32, ndim=3),
    mask: wp.array(dtype=wp.float32, ndim=3),
    depth: int,
    cum_dist: wp.array(dtype=wp.float32, ndim=3),
):
    y, x, d = wp.tid()

    s = float(0.0)
    for k in range(0, depth):
        s = s + buf_dist[y, x, k]

    cum_dist[y, x, d] = s * mask[y, x, d]


@wp.kernel
def apply_mask_to_light(
    buf_light: wp.array(dtype=wp.float32, ndim=3),
    mask: wp.array(dtype=wp.float32, ndim=3),
    buf_light_masked: wp.array(dtype=wp.float32, ndim=3),
):
    y, x, d = wp.tid()
    buf_light_masked[y, x, d] = buf_light[y, x, d] * mask[y, x, d]


@wp.kernel
def build_ray_paths(
    cum_dist: wp.array(dtype=wp.float32, ndim=3),
    buf_light_masked: wp.array(dtype=wp.float32, ndim=3),
    ray_path_wrt: wp.array(dtype=wp.float32, ndim=3),
    ray_path_prt: wp.array(dtype=wp.float32, ndim=3),
):
    y, x, d = wp.tid()
    ray_path_wrt[y, x, d] = 0.5 * (cum_dist[y, x, d] + buf_light_masked[y, x, d])
    ray_path_prt[y, x, d] = cum_dist[y, x, d]


import warp as wp
import numpy as np

wp.init()


# ------------------------------------------------------------
# 1) hard binning: equivalent to your Python ac_generate()
# ------------------------------------------------------------
@wp.kernel(enable_backward=False)
def ac_generate_hard_accumulate_slice(
    dis_3d: wp.array(dtype=wp.float32, ndim=3),
    img_3d: wp.array(dtype=wp.float32, ndim=3),
    slice_idx: int,
    height: int,
    lowlimit: float,
    resolution: float,
    length: int,
    flip_vertical: int,
    max_count: int,
    out_sum_image: wp.array(dtype=wp.float32, ndim=2),
    out_count: wp.array(dtype=wp.int32, ndim=2),
):
    y, x = wp.tid()

    yy = y
    if flip_vertical == 1:
        yy = height - 1 - y

    dval = dis_3d[yy, x, slice_idx]
    ival = img_3d[yy, x, slice_idx]

    dp = wp.int32(wp.floor((dval - lowlimit) / resolution))

    # keep exactly the same rule as your Python:
    # if dp >= length or dp <= 0: continue
    if dp > 0 and dp < length:
        old = wp.atomic_add(out_count, dp, x, 1)
        if old < max_count:
            wp.atomic_add(out_sum_image, dp, x, ival)


# ------------------------------------------------------------
# 2) median-of-runs AA: equivalent to your Python ac_generate_aa()
# ------------------------------------------------------------
@wp.kernel(enable_backward=False)
def ac_generate_aa_median_slice(
    dis_3d: wp.array(dtype=wp.float32, ndim=3),
    img_3d: wp.array(dtype=wp.float32, ndim=3),
    slice_idx: int,
    height: int,
    lowlimit: float,
    resolution: float,
    length: int,
    flip_vertical: int,
    max_segments: int,
    count_buffer: wp.array(dtype=wp.int32, ndim=2),
    out_sum_image: wp.array(dtype=wp.float32, ndim=2),
):
    y, x = wp.tid()

    yy = y
    if flip_vertical == 1:
        yy = height - 1 - y

    dval = dis_3d[yy, x, slice_idx]
    dp = wp.int32(wp.floor((dval - lowlimit) / resolution))

    # same validity rule as Python
    if dp > 0 and dp < length:

        # check whether current y is the start of a contiguous run
        is_run_start = 1
        if y > 0:
            yp = y - 1
            yyp = yp
            if flip_vertical == 1:
                yyp = height - 1 - yp

            dp_prev = wp.int32(wp.floor((dis_3d[yyp, x, slice_idx] - lowlimit) / resolution))
            if dp_prev == dp:
                is_run_start = 0

        if is_run_start == 1:
            # scan forward to find the end of this contiguous run
            end_y = y
            while end_y + 1 < height:
                yn = end_y + 1
                yyn = yn
                if flip_vertical == 1:
                    yyn = height - 1 - yn

                dp_next = wp.int32(wp.floor((dis_3d[yyn, x, slice_idx] - lowlimit) / resolution))

                if dp_next != dp:
                    break

                end_y = yn

            # Python find_medians():
            # odd  -> middle
            # even -> upper median
            median_y = y + (end_y - y + 1) // 2

            yym = median_y
            if flip_vertical == 1:
                yym = height - 1 - median_y

            old = wp.atomic_add(count_buffer, dp, x, 1)
            if old < max_segments:
                wp.atomic_add(out_sum_image, dp, x, img_3d[yym, x, slice_idx])


# ============================================================
# Helper: convert numpy to warp if needed
# ============================================================

def _to_wp_3d(x: Union[np.ndarray, wp.array], device):
    if isinstance(x, np.ndarray):
        return wp.array(x.astype(np.float32, copy=False), dtype=wp.float32, device=device)
    return x


# ============================================================
# Main stage: up to sum_image
# ============================================================

def process_buffers_to_sum_image(
    buf_dist: Union[np.ndarray, wp.array],
    buf_light: Union[np.ndarray, wp.array],
    buf_c: Union[np.ndarray, wp.array],
    uplimit: float,
    lowlimit: float,
    resolution: float,
    mask_eps: float = 0.01,
    use_mask: bool = True,
    flip_vertical: bool = True,
    device=None,
    return_numpy: bool = True,
):
    """
    Process buffers only up to sum_image.

    Inputs:
        buf_dist:  (H, W, D)
        buf_light: (H, W, D)
        buf_c:     (H, W, D)

    Returns:
        sum_image: (L, W), where L = floor((uplimit - lowlimit) / resolution)

    Notes:
        - noise removed
        - hard mask retained
        - ac_generate replaced by soft Gaussian splatting
        - only supports scalar buf_c
    """
    if device is None:
        device = wp.get_preferred_device()

    buf_dist_wp = _to_wp_3d(buf_dist, device)
    buf_light_wp = _to_wp_3d(buf_light, device)
    buf_c_wp = _to_wp_3d(buf_c, device)

    h, w, depth = buf_dist_wp.shape
    length = int(math.floor((uplimit - lowlimit) / resolution))

    if length <= 0:
        raise ValueError("Invalid sonar output length. Check uplimit/lowlimit/resolution.")

    # --------------------------------------------------------
    # mask
    # --------------------------------------------------------
    if use_mask:
        mask_wp = wp.zeros((h, w, depth), dtype=wp.float32, device=device)
        wp.launch(
            kernel=make_mask_from_dist,
            dim=(h, w, depth),
            inputs=[buf_dist_wp, float(mask_eps)],
            outputs=[mask_wp],
            device=device,
        )
    else:
        mask_wp = wp.ones((h, w, depth), dtype=wp.float32, device=device)

    # --------------------------------------------------------
    # cum_dist and masked light
    # --------------------------------------------------------
    cum_dist_wp = wp.zeros((h, w, depth), dtype=wp.float32, device=device)
    buf_light_masked_wp = wp.zeros((h, w, depth), dtype=wp.float32, device=device)

    wp.launch(
        kernel=reverse_cumsum_masked,
        dim=(h, w, depth),
        inputs=[buf_dist_wp, mask_wp, depth],
        outputs=[cum_dist_wp],
        device=device,
    )

    wp.launch(
        kernel=apply_mask_to_light,
        dim=(h, w, depth),
        inputs=[buf_light_wp, mask_wp],
        outputs=[buf_light_masked_wp],
        device=device,
    )

    # --------------------------------------------------------
    # ray_path_wrt / ray_path_prt
    # --------------------------------------------------------
    ray_path_wrt_wp = wp.zeros((h, w, depth), dtype=wp.float32, device=device)
    ray_path_prt_wp = wp.zeros((h, w, depth), dtype=wp.float32, device=device)

    wp.launch(
        kernel=build_ray_paths,
        dim=(h, w, depth),
        inputs=[cum_dist_wp, buf_light_masked_wp],
        outputs=[ray_path_wrt_wp, ray_path_prt_wp],
        device=device,
    )

    # --------------------------------------------------------
    # sum_image accumulation (new AA version)
    # use run-median AA instead of Gaussian accumulation
    # --------------------------------------------------------
    sum_image_wp = wp.zeros((length, w), dtype=wp.float32, device=device)

    flip_flag = 1 if flip_vertical else 0
    max_segments = 10000   # same role as Python ac_generate_aa()

    # WRT part: i = depth-1 ... 0
    for i in range(depth - 1, -1, -1):
        # scratch buffer for this slice only
        count_buffer_wp = wp.zeros((length, w), dtype=wp.int32, device=device)

        wp.launch(
            kernel=ac_generate_aa_median_slice,
            dim=(h, w),
            inputs=[
                ray_path_wrt_wp,
                buf_c_wp,
                i,
                h,
                float(lowlimit),
                float(resolution),
                length,
                flip_flag,
                int(max_segments),
            ],
            outputs=[
                count_buffer_wp,
                sum_image_wp,
            ],
            device=device,
        )

    # PRT part: i = depth-2 ... 0
    for i in range(depth - 1, 0, -1):
        # scratch buffer for this slice only
        count_buffer_wp = wp.zeros((length, w), dtype=wp.int32, device=device)

        wp.launch(
            kernel=ac_generate_aa_median_slice,
            dim=(h, w),
            inputs=[
                ray_path_prt_wp,
                buf_c_wp,
                i,
                h,
                float(lowlimit),
                float(resolution),
                length,
                flip_flag,
                int(max_segments),
            ],
            outputs=[
                count_buffer_wp,
                sum_image_wp,
            ],
            device=device,
        )

    if return_numpy:
        return sum_image_wp.numpy()

    return sum_image_wp


# ============================================================
# Quick smoke test
# ============================================================


def main():
    wp.init()
    scene_root = Path("marker_export")
    scene = load_scene_data(scene_root)
    buf_dist, buf_hitloc, buf_light, buf_c, buf_br = render_recursive(scene, depth=1)

    H, W, D = 720, 128, 3
    uplimit = 5.0
    lowlimit = 1.0
    resolution = 0.01

    sum_image = process_buffers_to_sum_image(
        buf_dist=buf_dist,
        buf_light=buf_light,
        buf_c=buf_c,
        uplimit=uplimit,
        lowlimit=lowlimit,
        resolution=resolution,
        mask_eps=0.01,
        use_mask=False,
        flip_vertical=True,
        return_numpy=True,
    )

    import matplotlib.pyplot as plt
    print("sum max:", float(np.max(sum_image)))
    print("sum min:", float(np.min(sum_image)))
    sum_image_log = 20.0 * np.log10(sum_image / 1000.0 + 1.0)
    print("log max:", float(np.max(sum_image_log)))
    print("log min:", float(np.min(sum_image_log)))
    sum_image_log_u8 = np.clip(sum_image_log / 70.0 * 255.0, 0, 255).astype(np.uint8)



    plt.figure(figsize=(6, 10))
    plt.imshow(sum_image_log_u8, cmap="gray", aspect="auto")
    plt.colorbar()
    plt.title("log10(sum_image)")
    plt.xlabel("azimuth / image width")
    plt.ylabel("range bin")
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    main()
