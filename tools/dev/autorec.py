#!/usr/bin/env python3
"""autorec.py : record every finished queue trial that is not yet in the segment journal; print the verdicts."""
import re, subprocess, sys
J = "/Users/nvm/Documents/projects/private/tensorfold-experiments/glm53-speed-seg4.jsonl"
SK = "/Users/nvm/.claude/skills/autonomous-experiment-loop/scripts/journal.py"
done_f = "/tmp/glm-trials/recorded.txt"
done = set(open(done_f).read().split()) if __import__("os").path.exists(done_f) else set()
cur = None
for line in open("/tmp/glm-trials/queue.log"):
    m = re.match(r"=== (\S+) \[(.*?)\]", line)
    if m:
        cur = (m.group(1), m.group(2)); continue
    m = re.match(r"LEAN (\S+) regime=(\S+) blocks=(\d+)/(\d+) POOLED ([\d.]+) prose ([\d.]+) code ([\d.]+)", line)
    if m and m.group(1) not in done:
        name, regime, _, _, pooled, prose, code = m.groups()
        env = cur[1] if cur and cur[0] == name else ""
        txt = f"{name}: {env} (prose {prose} code {code}, regime {regime})"
        out = subprocess.run(["python3", SK, "record", "--journal", J, "--mutation", txt, "--metric", pooled], capture_output=True, text=True).stdout
        print(name, pooled, prose, code, regime, "->", out.strip().splitlines()[0] if out.strip() else "?")
        done.add(name); open(done_f, "a").write(name + "\n")
        if "STOP" in out: print(out.strip().splitlines()[-1])
