import os

def get_initial_checkpoint(config, new_check_dir, name=None):
    checkpoint_dir = os.path.join("./logs", config.RECIPE_DIR, new_check_dir)
    if name:
        return os.path.join(checkpoint_dir, name)
    else:
        checkpoints = [checkpoint
                       for checkpoint in os.listdir(checkpoint_dir) if checkpoint.endswith('.pth')]
                    #    if checkpoint.startswith('top_') and checkpoint.endswith('.pth')]
        if checkpoints:
            print(list(sorted(checkpoints))[0])
            return os.path.join(checkpoint_dir, list(sorted(checkpoints))[0])
    return None
