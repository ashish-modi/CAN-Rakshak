from .base import NewFGSMAttack


class NewFGSMDoS(NewFGSMAttack):
    """
    DoS variant: injection (new synthetic frame) and modification (existing
    recorded frame) both perturb the ID and Data fields, with no projection
    to an existing arbitration ID.
    """

    attack_type = 'dos'
    default_target_id = '000'
