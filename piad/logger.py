import os
from PIL import Image

from tensorboardX import SummaryWriter
from torchvision.utils import make_grid


class Logger(object):
    def __init__(self, root):
        self.root = root
        self.image_dir = os.path.join(self.root, 'images')
        self.tensorboard_dir = os.path.join(self.root, 'tensorboard')

        os.makedirs(self.image_dir, exist_ok=True)
        os.makedirs(self.tensorboard_dir, exist_ok=True)

        self.writer = SummaryWriter(self.tensorboard_dir)

    def add_scalar(self, name, value, n_iter):
        self.writer.add_scalar(name, value, n_iter)

    def add_scalars(self, main_name, tag_value_dict, n_iter):
        self.writer.add_scalars(main_name, tag_value_dict, n_iter)

    def save_image_samples(self, images, grid_size, name, nrow=4):
        n = images.size(0)
        grid = make_grid(images, nrow=nrow)
        grid = (grid * 0.5 + 0.5) * 255
        grid = grid.clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
        grid = Image.fromarray(grid)
        grid = grid.resize((grid_size * nrow, grid_size * int(n / nrow)), Image.NEAREST)
        grid.save(os.path.join(self.image_dir, name))
