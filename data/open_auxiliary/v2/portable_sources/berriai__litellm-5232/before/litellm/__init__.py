verbose_logger = type('L', (), {'debug': lambda *a, **k: None, 'warning': lambda *a, **k: None})()
