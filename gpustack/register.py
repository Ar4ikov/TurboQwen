#!/usr/bin/env python3
"""Register this image as a GPUStack (v2.2+) custom backend and create a deployment with
its model route, through the management API.

  python3 gpustack/register.py --server http://192.168.1.71:8080 --session <jwt-or-cookie-file> \
      [--deployment gpustack/model-tp2-3090.json] [--replicas 1]

Three calls the UI makes for you and the raw API does not: the backend (from YAML), the
model, and the *model route* that makes the model visible to the /v1 gateway. A model
created through POST /v2/models alone answers on its backend port but the gateway says
"Model not found" until a route with a target on it exists. Every step is idempotent.
The session token is the gpustack_session cookie of an admin (a plain API key cannot
manage backends and models).
"""
import argparse, json, os, sys, urllib.error, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--server", required=True)
ap.add_argument("--session", required=True, help="admin session JWT, or a file containing it")
ap.add_argument("--backend", default=os.path.join(HERE, "backend.yaml"))
ap.add_argument("--deployment", default=os.path.join(HERE, "model-tp2-3090.json"))
ap.add_argument("--replicas", type=int, default=None, help="override the deployment's replica count")
ap.add_argument("--gpu", action="append", default=None, help="GPU id as GPUStack names it (<worker>:cuda:N); repeatable; overrides gpu_selector")
a = ap.parse_args()
tok = open(a.session).read().strip() if os.path.exists(a.session) else a.session
H = {"Cookie": f"gpustack_session={tok}", "Content-Type": "application/json"}

def call(method, path, body=None):
    req = urllib.request.Request(a.server.rstrip("/") + path, method=method, headers=H,
                                 data=json.dumps(body).encode() if body is not None else None)
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            raw = r.read().decode()
            return r.status, (json.loads(raw) if raw else None)
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()[:600]

def items(path):
    st, d = call("GET", path)
    return d.get("items", []) if isinstance(d, dict) else []

# 1. backend
yaml_text = open(a.backend, encoding="utf-8").read()
name = next(l.split(":", 1)[1].split("#")[0].strip() for l in yaml_text.splitlines() if l.startswith("backend_name:"))
backends = {b.get("backend_name"): b for b in items("/v2/inference-backends?perPage=100")}
if name in backends:
    print(f"backend {name}: exists (id {backends[name].get('id')})")
else:
    st, r = call("POST", "/v2/inference-backends/from-yaml", {"content": yaml_text})
    print(f"backend {name}: {st} {r if st != 200 else 'created id ' + str(r.get('id'))}")
    if st != 200:
        sys.exit(1)

# 2. model
dep = json.load(open(a.deployment, encoding="utf-8"))
if a.replicas is not None:
    dep["replicas"] = a.replicas
if a.gpu:
    dep["gpu_selector"] = {"gpu_ids": a.gpu, "gpus_per_replica": len(a.gpu)}
models = {m.get("name"): m for m in items("/v2/models?perPage=100")}
if dep["name"] in models:
    mid = models[dep["name"]]["id"]
    print(f"model {dep['name']}: exists (id {mid})")
else:
    st, r = call("POST", "/v2/models", dep)
    print(f"model {dep['name']}: {st} {r if st != 200 else 'created id ' + str(r.get('id'))}")
    if st != 200:
        sys.exit(1)
    mid = r["id"]

# 3. route (what the UI adds under the hood; the gateway resolves models through it)
routes = {r.get("name"): r for r in items("/v2/model-routes?perPage=100")}
if dep["name"] in routes:
    print(f"route {dep['name']}: exists (id {routes[dep['name']].get('id')})")
else:
    st, r = call("POST", "/v2/model-routes", {"name": dep["name"], "description": dep.get("description", ""),
                                             "categories": dep.get("categories", ["llm"]), "meta": {},
                                             "generic_proxy": False, "targets": [{"model_id": mid, "weight": 100}]})
    print(f"route {dep['name']}: {st} {r if st != 200 else 'created id ' + str(r.get('id'))}")
print("GPU ids on this cluster:", [g.get("id") for g in items("/v2/gpu-devices")])
print(f"scale with: PUT /v2/models/{mid} (replicas), then call the gateway as model={dep['name']}")
