"""The GPU on request (README "Releasing the GPU"), against the running stt_server.

  docker run --rm --network logus2k_network -e SPEECH_ADMIN_SECRET -v "$PWD/tests:/t:ro" -v <a 16 kHz mono wav>:/clip.wav:ro \
      --entrypoint python cortex:0.2 /t/release_check.py [http://stt_server:2700]

(any image with python-socketio[asyncio_client] and httpx). Checks: the admin routes refuse a missing or wrong secret;
a release frees the model's memory and says so; a transcription while released is served, with the same text, and
brings the model back; a renewed release; a release racing a transcription (both succeed, the text unchanged); load;
a lapsed lease brings the model back by itself (~80 s). Leaves the model loaded.
"""
import asyncio, os, sys, time, wave

import httpx
import socketio

URL = sys.argv[1] if len(sys.argv) > 1 else "http://stt_server:2700"
SECRET = os.environ["SPEECH_ADMIN_SECRET"]
H = {"X-Admin-Secret": SECRET}
failures = []


def check(ok, what, detail=""):
    print(f"{'ok  ' if ok else 'FAIL'} {what}" + (f": {detail}" if detail != "" else ""), flush=True)
    if not ok:
        failures.append(what)


w = wave.open("/clip.wav")
pcm16 = w.readframes(min(w.getnframes(), 12 * w.getframerate()))


async def main():
    a = httpx.AsyncClient(base_url=URL, timeout=120)
    check((await a.get("/admin/state")).status_code == 403, "no secret: refused")
    check((await a.get("/admin/state", headers={"X-Admin-Secret": "wrong"})).status_code == 403, "a wrong secret: refused")
    check((await a.get("/admin/nothing", headers=H)).status_code == 404, "an unknown route: 404")
    st = (await a.get("/admin/state", headers=H)).json()
    check(st["state"] == "loaded", "loaded at start", st)
    sio = socketio.AsyncClient(reconnection=False)
    await sio.connect(URL, transports=["websocket"], wait_timeout=10)
    say = lambda: sio.call("transcribe", {"audio": pcm16, "language": "en"}, timeout=120)
    ref = (await say())["text"]
    check(len(ref) > 20, "a transcription", ref[:80])

    r = (await a.post("/admin/release", headers=H, json={"reason": "check", "minutes": 5, "by": "release_check"})).json()
    check(r["state"] == "released" and r["freed_mib"] > 1000, "a release frees the model's memory", f"{r['freed_mib']} MiB, holds {r['gpu_mib']} MiB")
    check(r["lease"]["by"] == "release_check" and 290 <= r["lease"]["seconds_left"] <= 300, "with its lease", r["lease"])
    t = time.time(); txt = (await say())["text"]; dt = time.time() - t
    st = (await a.get("/admin/state", headers=H)).json()
    check(txt == ref, "a transcription while released: served, the same text", f"{dt:.2f} s")
    check(st["state"] == "loaded", "and the model is back", st["state"])

    r = (await a.post("/admin/release", headers=H, json={"reason": "check", "minutes": 5, "by": "release_check"})).json()
    r2 = (await a.post("/admin/release", headers=H, json={"reason": "check again", "minutes": 7, "by": "release_check"})).json()
    check(r2["state"] == "released" and r2["lease"]["seconds_left"] > 400 and r2["freed_mib"] == 0,
          "a second release renews the lease (nothing more to free)", r2["lease"])

    await a.post("/admin/load", headers=H)
    texts, rel = await asyncio.gather(
        asyncio.gather(*[say() for _ in range(3)]),
        a.post("/admin/release", headers=H, json={"reason": "race", "minutes": 5, "by": "release_check"}))
    check(all(x["text"] == ref for x in texts) and rel.status_code == 200, "a release racing transcriptions: all served, the same text",
          rel.json().get("state"))
    r = (await a.post("/admin/load", headers=H)).json()
    check(r["state"] == "loaded" and r["lease"]["until"] is None, "load: back, the lease closed", r["state"])
    check((await say())["text"] == ref, "and transcribing")

    r = (await a.post("/admin/release", headers=H, json={"reason": "lapse", "minutes": 1, "by": "release_check"})).json()
    check(r["state"] == "released", "a one-minute release")
    for _ in range(12):
        await asyncio.sleep(10)
        st = (await a.get("/admin/state", headers=H)).json()
        if st["state"] == "loaded":
            break
    check(st["state"] == "loaded" and st["lease"]["until"] is None, "a lapsed lease brings the model back by itself", st["state"])
    await sio.disconnect()

asyncio.run(main())
print(f"\n{len(failures)} failed" + (": " + "; ".join(failures) if failures else ""))
sys.exit(1 if failures else 0)
