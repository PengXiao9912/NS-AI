bias = True

model = dict(
    type='PinnNet',
    linear_module_cfgs=[
        dict(type='LinearModule',
             linear_cfg=dict(type='Linear', in_features=4, out_features=20, bias=bias),
             act_cfg=dict(type='ReLU')),
        dict(type='LinearModule',
             linear_cfg=dict(type='Linear', in_features=20, out_features=20, bias=bias),
             act_cfg=dict(type='ReLU')),
        dict(type='LinearModule',
             linear_cfg=dict(type='Linear', in_features=20, out_features=20, bias=bias),
             act_cfg=dict(type='ReLU')),
        dict(type='LinearModule',
             linear_cfg=dict(type='Linear', in_features=20, out_features=20, bias=bias),
             act_cfg=dict(type='ReLU')),
        dict(type='LinearModule',
             linear_cfg=dict(type='Linear', in_features=20, out_features=2, bias=bias),
             act_cfg=dict(type='ReLU')),
    ]
)
