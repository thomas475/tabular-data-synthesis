from framework.generators.ydata_synthetic.synthesizers.regular.cgan.model import CGAN
from framework.generators.ydata_synthetic.synthesizers.regular.wgan.model import WGAN
from framework.generators.ydata_synthetic.synthesizers.regular.vanillagan.model import VanilllaGAN
from framework.generators.ydata_synthetic.synthesizers.regular.wgangp.model import WGAN_GP
from framework.generators.ydata_synthetic.synthesizers.regular.dragan.model import DRAGAN
from framework.generators.ydata_synthetic.synthesizers.regular.cramergan.model import CRAMERGAN
from framework.generators.ydata_synthetic.synthesizers.regular.cwgangp.model import CWGANGP

__all__ = [
    "VanilllaGAN",
    "CGAN",
    "WGAN",
    "WGAN_GP",
    "DRAGAN",
    "CRAMERGAN",
    "CWGANGP"
]
