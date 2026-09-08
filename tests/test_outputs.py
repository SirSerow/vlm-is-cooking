import json
import unittest
from datetime import datetime, timezone
from urllib.request import urlopen

from kitchen_assistant.outputs import WebOutput
from kitchen_assistant.state import Frame
from kitchen_assistant.workflow import WorkflowView


class OutputTests(unittest.TestCase):
    def test_web_output_serves_state_and_latest_frame(self):
        output = WebOutput(port=0)
        output.start()
        try:
            view = WorkflowView(
                "Demo", "active", 0, "cut", "Cut", "Cut the onion.",
                ("knife",), ("onion",), (), (), "Cook", ("pan",), ("onion",),
                "in_progress", 0.8,
            )
            output.publish(Frame("one", datetime.now(timezone.utc), b"jpeg"), view)
            port = output._server.server_port
            with urlopen(f"http://127.0.0.1:{port}/state", timeout=2) as response:
                state = json.load(response)
            with urlopen(f"http://127.0.0.1:{port}/frame.jpg", timeout=2) as response:
                frame = response.read()
            self.assertEqual(state["step_id"], "cut")
            self.assertEqual(frame, b"jpeg")
        finally:
            output.close()


if __name__ == "__main__":
    unittest.main()

