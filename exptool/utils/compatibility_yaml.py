"""

compatibility_yaml.py

   backup option in case yaml is not installed

"""


def backup_yaml_parse(yamlin):
    """for absolute compatibility, a very very simple yaml reader
    built to purpose. strips all helpful yaml stuff out, so be careful!"""
    head_dict = dict()
    try:
        decoded = yamlin.decode()
    except:
        decoded = yamlin[0].decode()
    split = decoded.split('\n')
    #print(split)
    for k in split:
        #print(k)
        split2 = k.split(':')
        try:
            head_dict[split2[0].lstrip()] = split2[1].lstrip()
        except:
            pass
    return head_dict



