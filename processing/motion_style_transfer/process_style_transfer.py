from processing.motion_style_transfer.style_transfer.test import *

class StyleTransfer:
    def __init__(self, content_src, style_src, output_dir):
        self.content_src = content_src
        self.style_src = style_src
        self.output_dir = output_dir


    def apply_style_transfer(self):
        args = argparse.Namespace(name=None, batch_size=None, config='config', content_src=self.content_src, style_src=self.style_src, output_dir=self.output_dir)
        unique_id = main(args)

        return unique_id
        