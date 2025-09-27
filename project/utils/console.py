class C:
	R = "\033[31m"  # red
	G = "\033[32m"  # green
	Y = "\033[33m"  # yellow
	B = "\033[34m"  # blue
	M = "\033[35m"  # magenta
	C = "\033[36m"  # cyan
	D = "\033[0m"   # reset

def ok(msg: str) -> None:
	print(f"{C.G}[OK]	✔ {msg}{C.D}")

def fail(msg: str) -> None:
	print(f"{C.R}[FAIL]	✘ {msg}{C.D}")

def info(msg: str) -> None:
	print(f"{C.C}[INFO]	ℹ {C.D}{msg}")

def warn(msg: str) -> None:
	print(f"{C.Y}[WARN]	⚠ {msg}{C.D}")

def title(msg: str) -> None:
	print(f"\n{C.M}=== {msg} ==={C.D}")
