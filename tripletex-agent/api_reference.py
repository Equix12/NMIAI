"""Runtime API schema reference loaded from OpenAPI spec."""

import json
import os

_spec = None
_schemas = None


def _load_spec():
    global _spec, _schemas
    if _spec is None:
        spec_path = os.path.join(os.path.dirname(__file__), "..", "tripletex_openapi.json")
        if not os.path.exists(spec_path):
            spec_path = "/home/haava/NMAI/tripletex_openapi.json"
        with open(spec_path) as f:
            _spec = json.load(f)
        _schemas = _spec.get("components", {}).get("schemas", {})


def get_schema(entity_name: str) -> dict:
    """Get the full writable schema for an entity.

    Returns a dict with field names, types, and descriptions.
    Only includes fields that can be set on POST/PUT (excludes read-only).
    """
    _load_spec()

    # Try exact match first, then case-insensitive
    schema = _schemas.get(entity_name)
    if not schema:
        for k in _schemas:
            if k.lower() == entity_name.lower():
                schema = _schemas[k]
                entity_name = k
                break
    if not schema:
        return {"error": f"Schema '{entity_name}' not found"}

    props = schema.get("properties", {})
    result = {}
    for pname, pval in sorted(props.items()):
        if pname in ("id", "version", "url", "changes", "displayName"):
            continue
        if pval.get("readOnly"):
            continue

        ptype = pval.get("type", "")
        ref = pval.get("$ref", "")
        desc = pval.get("description", "")
        enum = pval.get("enum")

        field_info = {}
        if ref:
            ref_name = ref.split("/")[-1]
            field_info["type"] = f"object -> use {{\"id\": <int>}}"
            field_info["ref"] = ref_name
        elif ptype == "array":
            items = pval.get("items", {})
            items_ref = items.get("$ref", "")
            if items_ref:
                field_info["type"] = f"array of {items_ref.split('/')[-1]}"
            else:
                field_info["type"] = "array"
        else:
            field_info["type"] = ptype

        if desc:
            field_info["description"] = desc[:150]
        if enum:
            field_info["enum"] = enum

        result[pname] = field_info

    return {"entity": entity_name, "fields": result}


def get_endpoint_info(path: str, method: str = None) -> dict:
    """Get full endpoint info including parameters and request body schema.

    If method is not specified, returns info for all available methods on this path.
    """
    _load_spec()
    paths = _spec.get("paths", {})

    # Normalize path
    if not path.startswith("/"):
        path = "/" + path

    path_info = paths.get(path)
    if not path_info:
        # Try with /v2 prefix stripped or added
        for p in paths:
            if p.endswith(path) or path.endswith(p):
                path_info = paths[p]
                path = p
                break
    if not path_info:
        return {"error": f"No endpoint at {path}. Use search_api to find endpoints."}

    results = {}
    methods_to_check = [method.lower()] if method else [m for m in path_info if m != "parameters"]

    for m in methods_to_check:
        endpoint = path_info.get(m)
        if not endpoint:
            continue

        info = {
            "method": m.upper(),
            "path": path,
            "summary": endpoint.get("summary", ""),
            "description": endpoint.get("description", "")[:300],
        }

        # Query parameters
        params = endpoint.get("parameters", []) + path_info.get("parameters", [])
        if params:
            info["queryParameters"] = []
            for p in params:
                if p.get("in") == "query":
                    info["queryParameters"].append({
                        "name": p.get("name"),
                        "required": p.get("required", False),
                        "type": p.get("schema", {}).get("type", "string"),
                        "description": p.get("description", "")[:100],
                    })

        # Request body schema
        rb = endpoint.get("requestBody", {}).get("content", {})
        for ct, ct_val in rb.items():
            schema_ref = ct_val.get("schema", {}).get("$ref", "")
            if schema_ref:
                schema_name = schema_ref.split("/")[-1]
                info["requestBody"] = get_schema(schema_name)
                break

        results[m.upper()] = info

    if len(results) == 1:
        return list(results.values())[0]
    return {"path": path, "methods": results}


# Keep backward compat
def get_post_endpoint_info(path: str) -> dict:
    return get_endpoint_info(path, "post")


def search_endpoints(query: str) -> list:
    """Search for API endpoints matching a query string."""
    _load_spec()
    paths = _spec.get("paths", {})
    results = []
    query_lower = query.lower()

    for path, methods in paths.items():
        for method, details in methods.items():
            if method in ("parameters",):
                continue
            summary = details.get("summary", "")
            if query_lower in path.lower() or query_lower in summary.lower():
                results.append({
                    "method": method.upper(),
                    "path": path,
                    "summary": summary[:100],
                })

    return results[:20]
