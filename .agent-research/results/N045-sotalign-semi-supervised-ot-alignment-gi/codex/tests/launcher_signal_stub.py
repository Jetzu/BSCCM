#!/usr/bin/env python3
import os,signal,sys,time
from pathlib import Path
signal.signal(signal.SIGUSR1, lambda *args: sys.exit(75))
Path(os.environ["N045_READY"]).write_text("ready")
while True: time.sleep(.05)
