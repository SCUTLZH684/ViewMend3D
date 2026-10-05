"""CPU-only content identity for the Replica inputs used by the simulator."""
import hashlib
import json


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def scene_assets(upstream, mesh_path, stage_path):
    mesh, stage = (upstream / path for path in (mesh_path, stage_path))
    config = json.loads(stage.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not isinstance(config.get("render_asset"), str):
        raise ValueError("Stage configuration must name its render asset")
    paths = {mesh.resolve(), stage.resolve()}
    for name in ("render_asset", "semantic_asset", "nav_asset", "semantic_descriptor_filename"):
        if name in config:
            if not isinstance(config[name], str) or not config[name]:
                raise ValueError(f"Invalid stage asset: {name}")
            paths.add((stage.parent / config[name]).resolve())
    texture_dir = mesh.parent / "textures"
    textures = list(texture_dir.glob("*-color-ptex.hdr"))
    if not textures:
        raise ValueError("Replica PTex texture files are missing")
    paths.update(path.resolve() for path in textures)
    paths.add((texture_dir / "parameters.json").resolve())
    rows = []
    for path in sorted(paths):
        relative = path.relative_to(upstream.resolve()).as_posix()
        if not path.is_file() or not path.stat().st_size:
            raise ValueError(f"Missing/empty scene asset: {relative}")
        rows.append({"path": relative, "bytes": path.stat().st_size, "sha256": digest(path)})
    identity = hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return rows, identity
