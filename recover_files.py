import json
import os

transcript_path = r"C:\Users\ARTTESTT\.gemini\antigravity-ide\brain\95a06597-f3f8-466e-90f3-4ce047bda9b2\.system_generated\logs\transcript_full.jsonl"
workspace = r"c:\Users\ARTTESTT\Desktop\semantic-vector-main"

files_to_recover = [
    "visual_chain_env.py", "trPID_ENV.py", "test_headless.py", 
    "example_navigate.py", "vector.py", "spline.py", "trajectory.py"
]

recovered_files = {}

with open(transcript_path, 'r', encoding='utf-8') as f:
    for line in f:
        try:
            entry = json.loads(line)
            if 'tool_calls' in entry:
                # We can check if we wrote the file
                for tc in entry['tool_calls']:
                    if tc['name'] == 'default_api:write_to_file':
                        args = tc.get('arguments', {})
                        path = args.get('TargetFile', '')
                        filename = os.path.basename(path)
                        if filename in files_to_recover:
                            content = args.get('CodeContent', '')
                            # Overwrite with the latest version we wrote
                            recovered_files[filename] = content
            elif entry.get('type') == 'PLANNER_RESPONSE' or entry.get('type') == 'TOOL_RESPONSE':
                # Maybe view_file output is here
                content = entry.get('content', '')
                for filename in files_to_recover:
                    if f"File Path: `file:///" in content and filename in content:
                        # simple heuristic to find file dumps in transcript
                        parts = content.split(f"File Path: `file:///")
                        for part in parts:
                            if filename in part and "Total Lines:" in part:
                                # extract code
                                lines = part.split('\n')
                                code_lines = []
                                start_collecting = False
                                for l in lines:
                                    if "The following code has been modified" in l:
                                        start_collecting = True
                                        continue
                                    if "The above content shows the entire" in l:
                                        break
                                    if start_collecting:
                                        # Remove line numbers: "1: import cv2" -> "import cv2"
                                        if ": " in l:
                                            code_lines.append(l.split(": ", 1)[1])
                                        else:
                                            code_lines.append(l)
                                if code_lines:
                                    recovered_files[filename] = '\n'.join(code_lines)
                                    
        except Exception as e:
            pass

for filename, content in recovered_files.items():
    out_path = os.path.join(workspace, filename)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"Recovered {filename}")
