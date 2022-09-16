from terminaltables import AsciiTable


def print_info(cfg):
    model = cfg.pop('model')

    # pretrained = os.path.basename(cfg.get('train').get('pretrained_weights')) if cfg.get('train').get('pretrained_flag') else 'None'
    # freeze = ' '.join(list(cfg.get('train').get('freeze_layers')))

    TITLE = 'Model info'
    TABLE_DATA = (
        ('model', 'cfg'),
        (model, cfg))

    table_instance = AsciiTable(TABLE_DATA, TITLE)
    print()
    print(table_instance.table)
    print()

def print_train_val_info(train_lose,val_lose,train_acc,val_acc):
    # model = cfg.pop('model')


    # pretrained = os.path.basename(cfg.get('train').get('pretrained_weights')) if cfg.get('train').get('pretrained_flag') else 'None'
    # freeze = ' '.join(list(cfg.get('train').get('freeze_layers')))

    TITLE = 'train_val info'
    TABLE_DATA = (
        ('train_lose', 'val_lose','train_acc','val_acc'),
        (train_lose, val_lose,train_acc,val_acc))

    table_instance = AsciiTable(TABLE_DATA, TITLE)
    print()
    print(table_instance.table)
    print()