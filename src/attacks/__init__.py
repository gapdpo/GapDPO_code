from .gcg import GCG
from .gcq import GCQ
from .beast import BEAST
from .pair import PAIR
from .autodan_zhu import AutoDANZhu
from .conti_embed import ContiEmbed


__attacker_zoo__ = {
    "gcg"         : GCG,
    "gcq"         : GCQ,
    "beast"       : BEAST,
    "pair"        : PAIR,
    "autodan-zhu" : AutoDANZhu,
    "conti-embed" : ContiEmbed,
}

def build_attacker(name: str, **kwargs):
    return __attacker_zoo__[name](**kwargs)
