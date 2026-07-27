# Local Quest TLS files

The Quest/Vuer bridge expects `cert.pem` and `key.pem` in this directory.
They are machine-local credentials and are intentionally not copied from the
source repository or tracked here.

Before running teleoperation, either copy your existing development
certificate and key into this directory or generate a new pair trusted by the
Quest/browser. Keep `key.pem` private.
