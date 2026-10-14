"""Only exact OLD alignment geometry; no empirical SAD/CCORR thresholds."""
def aligned_pairs(model, gray):
    radius = model.model['translation_radius']
    for name, template in model.templates:
        if template.shape != gray.shape:
            continue
        height, width = gray.shape
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                a, b = max(0, -dx), max(0, -dy)
                A, B = min(width, width - dx), min(height, height - dy)
                yield name, dx, dy, template[b:B, a:A], gray[b+dy:B+dy, a+dx:A+dx]
