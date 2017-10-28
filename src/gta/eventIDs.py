keys2names = {}

## GAMEPAD BUTTONS
# x360ce.ini assigns int or str keys to each button press.
keys2names['gamepad buttons'] = {
    1:  'gamepad_d_up',     2: 'gamepad_d_down', 3: 'gamepad_d_left', 4: 'gamepad_d_right',
    5:  'gamepad_start',    6: 'gamepad_back',
    7:  'gamepad_l_stick',  8: 'gamepad_r_stick',
    9:  'gamepad_l_bumper', 10: 'gamepad_r_bumper',
    13: 'gamepad_a',        14: 'gamepad_b',
    15: 'gamepad_x',        16: 'gamepad_y',
}

## GAMEPAD AXES
keys2names['gamepad axes'] = {
    k: 'gamepad_%s' % k
    for k in [
        'l_thumb_x',
        'l_thumb_y',
        'r_thumb_x',
        'r_thumb_y',
        'left_trigger',
        'right_trigger',
    ]
}

## KEYBOARD KEYS
keyNames = [
    'left shift', 'right shift', 
    'left ctrl', 'right ctrl', 
    'left alt', 'right alt',
    'enter', 'space', 'backspace',
    'page up', 'page down', 'home', 'end', 'insert', 'delete',
    'up', 'down', 'left', 'right',
    'caps lock', 'tab', 'menu', 'esc', 'num lock',
    'print screen', 'scroll lock', 'pause'
]
keyNames.extend(['f%d' % i for i in range(1, 13)])
keyNames.extend(''.join([s.strip() for s in '''
    abcdefghijklmnopqrstuvwxyz
    `1234567890-=
    ~!@#$%^&*()_+
    []\{\}
'''.split()]))
_ = list('1234567890/*-+')
_.extend([
    'enter', 'decimal',
    'home', 'end', 'page up', 'page down',
    'up', 'down', 'left', 'right',
    'insert', 'delete',
    'clear', # 5_keypad without num lock
])
keyNames.extend([k+'_keypad' for k in _])
keys2names['keyboard keys'] = {
    k : 'keyboard_%s' % k
    for k in keyNames
}

# Allow for converting any key to its labeling name.
keys2names['all'] = {}
names2eids = {}
idNum = 0
for category in sorted(keys2names.keys()):
    if category != 'all':
        keys2names['all'].update(keys2names[category])
        for name in sorted(keys2names[category].values()):
            names2eids[name] = idNum
            idNum += 1

# Precompute other transformations.
_reverseDict = lambda d: {v: k  for (k, v) in d.items()}

eids2names = _reverseDict(names2eids)
names2keys = _reverseDict(keys2names['all'])
keys2eids = {key: names2eids[name] for (key, name) in keys2names['all'].items()}
eids2keys = {eid: names2keys[name] for (name, eid) in names2eids.items()}

# Differentiate a few specific classes of controls which I might want to treat differently.
# E.g., axis states are float, but button states are bool. 
# And we might not want to consider keyboard input.
axisEids = [
    keys2eids[key]
    for key in keys2names['gamepad axes'].keys()
]
axisEids.sort()
keyboardEids = [
    keys2eids[key]
    for key in keys2names['keyboard keys'].keys()
]
keyboardEids.sort()
gamepadButtonEids = [
    keys2eids[key]
    for key in keys2names['gamepad buttons'].keys()
]
gamepadButtonEids.sort()
gamepadEids = [eid for eid in keys2eids.values() if eid not in keyboardEids]
gamepadEids.sort()

# Delete some temporary variables.
del name, keyNames, idNum, category
