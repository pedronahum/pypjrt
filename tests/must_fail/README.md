# must_fail

Each `*_must_fail.py` here **must be rejected by pyright**. They are the only
form of "compile error" Python offers, and the reason they exist is to stop the
typed-façade claim drifting ahead of what the checker actually proves

Never "fix" one into checking cleanly — that deletes the guarantee. If a probe
starts passing, either the façade regressed or the claim was wrong.

Borrowed from the `MustFail/` convention: a script compiles each probe and fails if any
of them *succeeds*.
