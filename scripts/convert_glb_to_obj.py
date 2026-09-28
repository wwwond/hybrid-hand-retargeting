import trimesh
from pathlib import Path

MESH_DIR = Path.home() / "hand-retarget/dex-retargeting-repo/assets/robots/hands/inspire_hand"

glb_files = list(MESH_DIR.rglob("*.glb"))
print(f"Found {len(glb_files)} .glb files")

failed = []
for glb_path in glb_files:
    obj_path = glb_path.with_suffix(".obj")
    if obj_path.exists():
        continue
    try:
        scene = trimesh.load(glb_path)
        if isinstance(scene, trimesh.Scene):
            mesh = scene.dump(concatenate=True)  # applies node transform
        else:
            mesh = scene
        mesh.export(obj_path)
        print(f"  OK: {glb_path.name} -> {obj_path.name}")
    except Exception as e:
        print(f"  FAIL: {glb_path.name}  ({e})")
        failed.append(glb_path.name)

print(f"\nDone. {len(failed)} failed: {failed}")
