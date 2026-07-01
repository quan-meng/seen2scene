from blenderproc.python.loader.Front3DLoader import _Front3DLoader as BaseFront3DLoader
from blenderproc.python.loader.ObjectLoader import load_obj
from blenderproc.python.utility.LabelIdMapping import LabelIdMapping
from blenderproc.python.utility.Utility import resolve_path
from blenderproc.python.types.MeshObjectUtility import MeshObject
import bpy
import numpy as np
import json
import trimesh
import os
import warnings
from typing import List, Optional
from mathutils import Matrix


class _Front3DLoader(BaseFront3DLoader):
    @staticmethod
    def load_furniture_objs(
        data: dict,
        future_model_path: str,
        lamp_light_strength: float,
        label_mapping: LabelIdMapping,
    ) -> List[MeshObject]:
        """
        Load all furniture objects specified in the json file, these objects are stored as "raw_model.obj" in the
        3D_future_model_path. For lamp the lamp_light_strength value can be changed via the config.

        :param data: json data dir. Should contain "furniture"
        :param future_model_path: Path to the models used in the 3D-Front dataset.
        :param lamp_light_strength: Strength of the emission shader used in each lamp.
        :param label_mapping: A dict which maps the names of the objects to ids.
        :return: The list of loaded mesh objects.
        """
        # collect all loaded furniture objects
        all_objs = []
        # for each furniture element
        for ele in data["furniture"]:
            # create the paths based on the "jid"
            folder_path = os.path.join(future_model_path, ele["jid"])
            obj_file = os.path.join(folder_path, "raw_model.obj")
            # if the object exists load it -> a lot of object do not exist
            # we are unsure why this is -> we assume that not all objects have been made public
            if (
                os.path.exists(obj_file)
                and not "7e101ef3-7722-4af8-90d5-7c562834fabd" in obj_file
            ):
                # load all objects from this .obj file
                objs = load_obj(filepath=obj_file)
                # extract the name, which serves as category id
                used_obj_name = ""
                if "category" in ele:
                    used_obj_name = ele["category"]
                elif "title" in ele:
                    used_obj_name = ele["title"]
                    if "/" in used_obj_name:
                        used_obj_name = used_obj_name.split("/")[0]
                if used_obj_name == "":
                    used_obj_name = "others"
                for obj in objs:
                    obj.set_name(used_obj_name)
                    # add some custom properties
                    obj.set_cp("uid", ele["uid"])
                    obj.set_cp("jid", ele["jid"])
                    # this custom property determines if the object was used before
                    # is needed to only clone the second appearance of this object
                    obj.set_cp("is_used", False)
                    obj.set_cp("is_3D_future", True)
                    obj.set_cp(
                        "3D_future_type", "Non-Object"
                    )  # is an non object used for the interesting score
                    # set the category id based on the used obj name
                    used_obj_name = (
                        used_obj_name.lower() if used_obj_name != "unknown" else "void"
                    )
                    obj.set_cp(
                        "category_id",
                        label_mapping.id_from_label(used_obj_name.lower()),
                    )
                    # walk over all materials
                    for mat in obj.get_materials():
                        if mat is None:
                            continue
                        principled_node = mat.get_nodes_with_type("BsdfPrincipled")
                        if (
                            "bed" in used_obj_name.lower()
                            or "sofa" in used_obj_name.lower()
                        ):
                            if len(principled_node) == 1:
                                principled_node[0].inputs[
                                    "Roughness"
                                ].default_value = 0.5
                        is_lamp = "lamp" in used_obj_name.lower()
                        if len(principled_node) == 0 and is_lamp:
                            # this material has already been transformed
                            continue
                        if len(principled_node) == 1:
                            principled_node = principled_node[0]
                        else:
                            raise ValueError(
                                f"The amount of principle nodes can not be more than 1, "
                                f"for obj: {obj.get_name()}!"
                            )

                        # Front3d .mtl files contain emission color which make the object mistakenly emissive
                        # => Reset the emission color
                        if "Emission" in principled_node.inputs:
                            principled_node.inputs["Emission"].default_value[:3] = [
                                0,
                                0,
                                0,
                            ]

                        # Front3d .mtl files use Tf incorrectly, they make all materials fully transmissive
                        # Revert that:
                        # Blender 4.0+ uses "Transmission Weight", older versions use "Transmission"
                        if "Transmission Weight" in principled_node.inputs:
                            principled_node.inputs[
                                "Transmission Weight"
                            ].default_value = 0
                        elif "Transmission" in principled_node.inputs:
                            principled_node.inputs["Transmission"].default_value = 0

                        # For each a texture node
                        image_node = mat.new_node("ShaderNodeTexImage")
                        # and load the texture.png
                        base_image_path = os.path.join(folder_path, "texture.png")
                        image_node.image = bpy.data.images.load(
                            base_image_path, check_existing=True
                        )
                        mat.link(
                            image_node.outputs["Color"],
                            principled_node.inputs["Base Color"],
                        )
                        # if the object is a lamp, do the same as for the ceiling and add an emission shader
                        if is_lamp:
                            mat.make_emissive(lamp_light_strength)

                all_objs.extend(objs)
            elif "7e101ef3-7722-4af8-90d5-7c562834fabd" in obj_file:
                warnings.warn(
                    f"This file {obj_file} was skipped as it can not be read by blender."
                )
        return all_objs


def load_front3d(
    json_path: str,
    future_model_path: str,
    front_3D_texture_path: str,
    label_mapping: LabelIdMapping,
    ceiling_light_strength: float = 0.8,
    lamp_light_strength: float = 7.0,
    black_filter_path: Optional[str] = None,
) -> List[MeshObject]:
    """Loads the 3D-Front scene specified by the given json file.

    :param json_path: Path to the json file, where the house information is stored.
    :param future_model_path: Path to the models used in the 3D-Front dataset.
    :param front_3D_texture_path: Path to the 3D-FRONT-texture folder.
    :param label_mapping: A dict which maps the names of the objects to ids.
    :param ceiling_light_strength: Strength of the emission shader used in the ceiling.
    :param lamp_light_strength: Strength of the emission shader used in each lamp.
    :return: The list of loaded mesh objects.
    """
    json_path = resolve_path(json_path)
    future_model_path = resolve_path(future_model_path)
    front_3D_texture_path = resolve_path(front_3D_texture_path)

    if not os.path.exists(json_path):
        raise FileNotFoundError(f"The given path does not exists: {json_path}")
    if not json_path.endswith(".json"):
        raise FileNotFoundError(
            f"The given path does not point to a .json file: {json_path}"
        )
    if not os.path.exists(future_model_path):
        raise FileNotFoundError(
            f"The 3D future model path does not exist: {future_model_path}"
        )

    # load data from json file
    with open(json_path, "r", encoding="utf-8") as json_file:
        data = json.load(json_file)

    if "scene" not in data:
        raise ValueError(f"There is no scene data in this json file: {json_path}")

    structure_objects = _Front3DLoader.create_mesh_objects_from_file(
        data, front_3D_texture_path, ceiling_light_strength, label_mapping, json_path
    )
    furniture_objects = _Front3DLoader.load_furniture_objs(
        data, future_model_path, lamp_light_strength, label_mapping
    )

    if black_filter_path is not None:
        scene_name = os.path.basename(json_path).split(".")[0]
        with open(black_filter_path, "r") as json_file:
            black_filters = json.load(json_file)

        if scene_name in black_filters:
            furniture_objects = [
                x
                for x in furniture_objects
                if x.get("jid") not in black_filters[scene_name]
            ]

    furniture_objects = _Front3DLoader.move_and_duplicate_furniture(
        data, furniture_objects
    )

    # add an identifier to the obj
    for obj in structure_objects + furniture_objects:
        obj.set_cp("is_3d_front", True)

    return structure_objects, furniture_objects


# from filelock import FileLock
def load_black_dict(file_path: str) -> List[str]:
    with open(file_path, "r") as file:
        back_list = file.read().splitlines()
    back_dict = {}
    for back_str in back_list:
        scene_name, object_jid = back_str.split(",")
        if scene_name not in back_dict:
            back_dict[scene_name] = []
        else:
            back_dict[scene_name].append(object_jid)
    return back_dict


def export_tri_scene_mesh(
    objects,
    scene_name: str = "",
    black_filter_path: Optional[str] = None,
    out_dir: Optional[str] = None,
    filter_flags: Optional[List[str]] = None,
) -> trimesh.Trimesh:
    loaded_objects, is_triangles = [], []
    metadata = {"object_bboxes": [], "object_categories": [], "object_names": []}
    scene_box = np.array([[np.inf] * 3, [-np.inf] * 3])

    for mesh_object in objects:
        name = mesh_object.get_name()

        mesh_bpy = mesh_object.get_mesh().copy()
        mesh_bpy.transform(Matrix(mesh_object.get_local2world_mat()))
        vertices = np.array([list(v.co) for v in mesh_bpy.vertices])

        if "empty" in filter_flags and len(vertices) == 0:
            print(
                f"Filter empty: No vertices found in object: {name} in scene: {scene_name}"
            )
            # Clean up temporary mesh copy
            bpy.data.meshes.remove(mesh_bpy)
            continue

        # Skip objects with NaN vertices
        if "nan" in filter_flags and np.isnan(vertices).any():
            print(
                f"Filter nan: NaN values found in vertices of object: {name} in scene: {scene_name}"
            )
            # Clean up temporary mesh copy
            bpy.data.meshes.remove(mesh_bpy)
            continue

        # Skip objects with center too far from origin
        bound_max = np.max(vertices, axis=0)
        bound_min = np.min(vertices, axis=0)
        obj_center = (bound_max + bound_min) / 2
        if "bound" in filter_flags and np.max(np.abs(obj_center)) > 40.0:
            print(f"Filter far bound: Object center too far from origin: {name}")
            # Clean up temporary mesh copy
            bpy.data.meshes.remove(mesh_bpy)
            continue

        # Skip objects with size > 10
        object_size = bound_max - bound_min
        if "size" in filter_flags:
            if object_size[0] > 10.0 or object_size[1] > 10.0 or object_size[2] > 5.0:
                print(
                    f"Filter large size: Object size too large: {object_size} in scene: {scene_name}"
                )
                # Clean up temporary mesh copy
                bpy.data.meshes.remove(mesh_bpy)
                continue

        # Skip objects with vertices too far from origin
        max_vertex = np.max(np.abs(vertices))
        if max_vertex > 100.0:
            print(
                f"Filter large vertex: Object contains vertices with large values: {max_vertex} in scene: {scene_name}"
            )
            # Clean up temporary mesh copy
            bpy.data.meshes.remove(mesh_bpy)
            continue

        is_triangle = all(
            len(list(face.vertices[:])) == 3 for face in mesh_bpy.polygons
        )
        is_triangles.append(is_triangle)
        loaded_objects.append(mesh_object)

        # Store metadata
        object_bbox = mesh_object.get_bound_box()  # [8, 3]
        bbox_mins = np.min(object_bbox, axis=0)  # [3]
        bbox_maxs = np.max(object_bbox, axis=0)  # [3]
        object_bbox = np.stack([bbox_mins, bbox_maxs])  # [2, 3]
        metadata["object_bboxes"].append(object_bbox.tolist())
        metadata["object_categories"].append(mesh_object.get_cp("category_id"))
        metadata["object_names"].append(name)

        scene_box[0] = np.minimum(scene_box[0], np.min(vertices, axis=0))
        scene_box[1] = np.maximum(scene_box[1], np.max(vertices, axis=0))
        scene_box[0][-1] = 0.0  # floor at z=0

        # Clean up temporary mesh copy after use
        bpy.data.meshes.remove(mesh_bpy)

    return loaded_objects, is_triangles, metadata, scene_box
