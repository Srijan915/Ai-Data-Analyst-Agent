"""Optional real-browser smoke/capture: uses local Chrome and installed websockets.

No provider calls: uploads the public sample, confirms a key, captures actual UI.
Run from the project root: python scripts/capture_ui.py
"""
import base64
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request

from websockets.sync.client import connect

ROOT = Path(__file__).resolve().parents[1]


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def main():
    chrome = next((p for p in (
        Path("C:/Program Files/Google/Chrome/Application/chrome.exe"),
        Path("C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe"),
    ) if p.exists()), None)
    if chrome is None:
        raise SystemExit("Install Chrome/Edge or capture screenshots manually as described in README.")
    env = dict(os.environ, PYTHON_DOTENV_DISABLED="1", APP_ENV="development", APP_ACCESS_PASSWORD="")
    # This check deliberately does not use provider credentials, even if supplied.
    env.pop("GEMINI_API_KEY", None)
    app_port, debug_port = free_port(), free_port()
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    processes = []
    with tempfile.TemporaryDirectory(prefix="analyst-browser-") as profile:
        try:
            processes.append(subprocess.Popen([
                sys.executable, "-m", "streamlit", "run", "app/main.py",
                "--server.headless=true", "--server.address=127.0.0.1",
                f"--server.port={app_port}", "--browser.gatherUsageStats=false",
            ], cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=flags))
            for _ in range(60):
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{app_port}/_stcore/health", timeout=1) as response:
                        if response.read() == b"ok":
                            break
                except OSError:
                    time.sleep(.25)
            else:
                raise RuntimeError("Streamlit did not become healthy")
            processes.append(subprocess.Popen([
                str(chrome), "--headless=new", "--disable-gpu", "--no-first-run",
                "--no-default-browser-check", f"--user-data-dir={profile}",
                f"--remote-debugging-port={debug_port}", "--remote-debugging-address=127.0.0.1",
                "about:blank",
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=flags))
            for _ in range(60):
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{debug_port}/json", timeout=1) as response:
                        target = next(t for t in json.load(response) if t["type"] == "page")
                    break
                except (OSError, StopIteration):
                    time.sleep(.25)
            else:
                raise RuntimeError("Browser did not become ready")
            with connect(target["webSocketDebuggerUrl"], max_size=16_000_000) as ws:
                message_id = 0

                def call(method, **params):
                    nonlocal message_id
                    message_id += 1
                    ws.send(json.dumps(dict(id=message_id, method=method, params=params)))
                    while True:
                        response = json.loads(ws.recv(timeout=20))
                        if response.get("id") == message_id:
                            if "error" in response:
                                raise RuntimeError(response["error"])
                            return response.get("result", {})

                def js(expression):
                    result = call("Runtime.evaluate", expression=expression, returnByValue=True)
                    if "exceptionDetails" in result:
                        raise RuntimeError(result["exceptionDetails"])
                    return result["result"].get("value")

                def wait_for(expression):
                    for _ in range(100):
                        if js(expression):
                            return
                        time.sleep(.2)
                    raise RuntimeError("UI condition timed out: " + expression)

                def screenshot(name):
                    time.sleep(.6)
                    content = call("Page.captureScreenshot", format="png", captureBeyondViewport=False)
                    (ROOT / "screenshots" / name).write_bytes(base64.b64decode(content["data"]))

                call("Page.enable")
                call("Emulation.setDeviceMetricsOverride", width=1440, height=1000, deviceScaleFactor=1, mobile=False)
                call("Page.navigate", url=f"http://127.0.0.1:{app_port}")
                wait_for("document.body.innerText.includes('Upload a CSV, XLSX, TXT or PDF file above')")
                assert not js("document.body.innerText.includes('Python executable:')")
                screenshot("streamlit-home.png")
                document = call("DOM.getDocument")
                node = call("DOM.querySelector", nodeId=document["root"]["nodeId"], selector='input[type="file"]')
                call("DOM.setFileInputFiles", nodeId=node["nodeId"], files=[
                    str(ROOT / "data/sample/customer_shopping_behavior.csv"),
                    str(ROOT / "data/sample/item_catalog.csv"),
                ])
                wait_for("document.body.innerText.includes('Confirm join')")
                js("[...document.querySelectorAll('button')].find(b=>b.innerText==='Confirm join').click()")
                wait_for("document.body.innerText.includes('Revoke join')")
                assert not js("document.body.innerText.includes('Traceback')")
                js("document.querySelector('[data-testid=stMain]').scrollTop=0")
                screenshot("streamlit-workspace.png")
                js("[...document.querySelectorAll('h3')].find(h=>h.innerText.includes('Relationships')).scrollIntoView({block:'start'})")
                screenshot("streamlit-relationships.png")
                call("Emulation.setDeviceMetricsOverride", width=1280, height=800, deviceScaleFactor=1, mobile=False)
                js("[...document.querySelectorAll('h3')].find(h=>h.innerText.includes('Ask a question')).scrollIntoView({block:'start'})")
                screenshot("streamlit-question-laptop.png")
                assert js("document.querySelector('textarea').getBoundingClientRect().width > 200")
                assert js("document.documentElement.scrollWidth <= window.innerWidth")
                # Verify a real pre-provider refusal with the uploaded files.
                js("document.querySelector('textarea').focus()")
                call("Input.insertText", text="What is the monthly purchase trend over order dates?")
                call("Input.dispatchKeyEvent", type="keyDown", key="Tab", code="Tab", windowsVirtualKeyCode=9)
                call("Input.dispatchKeyEvent", type="keyUp", key="Tab", code="Tab", windowsVirtualKeyCode=9)
                time.sleep(.5)
                js("[...document.querySelectorAll('button')].find(b=>b.innerText==='Analyze question').click()")
                wait_for("document.body.innerText.includes('date/time') || document.body.innerText.includes('date or time')")
                wait_for("![...document.querySelectorAll('button')].some(b=>b.innerText==='Stop')")
                call("Emulation.setDeviceMetricsOverride", width=390, height=844, deviceScaleFactor=1, mobile=True)
                js("document.querySelector('[data-testid=stMain]').scrollTop=0")
                screenshot("streamlit-mobile.png")
                assert js("document.documentElement.scrollWidth <= window.innerWidth")
                assert not js("document.body.innerText.includes('Antigravity')")
                print("PASS: real browser upload, confirmation, desktop/mobile layout, missing-date refusal, screenshots; no provider calls.")
                call("Browser.close")
        finally:
            for process in reversed(processes):
                if process.poll() is None:
                    process.terminate()
                process.wait(timeout=10)


if __name__ == "__main__":
    main()
