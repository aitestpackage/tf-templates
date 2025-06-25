# logger_ui.py
from IPython.display import display, HTML
from .client import send_message, fetch_messages
import uuid

def init_logger(user="sys"):
    user_id = f"{user}_{uuid.uuid4().hex[:6]}"
    display(HTML(f"""
    <div id="logger-box" style="font-size:smaller;color:#555;margin-top:1em">
        <div><b>System Logger (runtime)</b></div>
        <pre id="log-output" style="white-space:pre-wrap;max-height:160px;overflow:auto;
             border:1px solid #ddd;padding:6px;background:#fcfcfc;border-radius:6px;
             font-family:monospace; font-size:11px;"></pre>
        <input type="text" id="log-input" placeholder="Enter log message..." style="width:75%;font-size:small">
        <button onclick="sendLog()" style="font-size:x-small">Log</button>
    </div>
    <script>
    async function sendLog() {{
        let msg = document.getElementById("log-input").value;
        await fetch("{SERVER_URL}/send", {{
            method: "POST",
            headers: {{'Content-Type': 'application/json'}},
            body: JSON.stringify({{user: "{user_id}", msg: msg}})
        }});
        document.getElementById("log-input").value = '';
        updateLogs();
    }}
    async function updateLogs() {{
        let res = await fetch("{SERVER_URL}/recv");
        let data = await res.json();
        document.getElementById("log-output").textContent = data.msg || "(no logs)";
    }}
    setInterval(updateLogs, 3000);
    updateLogs();
    </script>
    """))
