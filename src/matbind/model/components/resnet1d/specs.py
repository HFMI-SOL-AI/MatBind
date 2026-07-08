RESNET_SPECS = {
    777: [
        ("residual", 64, 1),
        ("residual", 128, 15),
        ("residual", 256, 15),
        ("residual", 256, 15),
        ("residual", 512, 15),
        ("residual", 512, 1),
    ],
    666: [
        ("residual", 64, 3),
        ("residual", 128, 4),
        ("residual", 128, 13),
        ("residual", 256, 20),
        ("residual", 256, 30),
        ("residual", 512, 3),
    ],
    10: [
        ("residual", 64, 1),
        ("residual", 128, 1),
        ("residual", 256, 1),
        ("residual", 512, 1),
    ],
    18: [
        ("residual", 64, 2),
        ("residual", 128, 2),
        ("residual", 256, 2),
        ("residual", 512, 2),
    ],
    34: [
        ("residual", 64, 3),
        ("residual", 128, 4),
        ("residual", 256, 6),
        ("residual", 512, 3),
    ],
    50: [
        ("bottleneck", 64, 3),
        ("bottleneck", 128, 4),
        ("bottleneck", 256, 6),
        ("bottleneck", 512, 3),
    ],
    101: [
        ("bottleneck", 64, 3),
        ("bottleneck", 128, 4),
        ("bottleneck", 256, 23),
        ("bottleneck", 512, 3),
    ],
    152: [
        ("bottleneck", 64, 3),
        ("bottleneck", 128, 8),
        ("bottleneck", 256, 36),
        ("bottleneck", 512, 3),
    ],
    200: [
        ("bottleneck", 64, 3),
        ("bottleneck", 128, 24),
        ("bottleneck", 256, 36),
        ("bottleneck", 512, 3),
    ],
    270: [
        ("bottleneck", 64, 4),
        ("bottleneck", 128, 29),
        ("bottleneck", 256, 53),
        ("bottleneck", 512, 4),
    ],
    350: [
        ("bottleneck", 64, 4),
        ("bottleneck", 128, 36),
        ("bottleneck", 256, 72),
        ("bottleneck", 512, 4),
    ],
    420: [
        ("bottleneck", 64, 4),
        ("bottleneck", 128, 44),
        ("bottleneck", 256, 87),
        ("bottleneck", 512, 4),
    ],
}

DECONV_RESNET_SPECS = {
    10: [
        ("residual", 512, 1),
        ("residual", 256, 1),
        ("residual", 128, 1),
        ("residual", 64, 1),
    ]
}
