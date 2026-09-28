from .base import NewFGSMAttack


class NewFGSMSpoof(NewFGSMAttack):
    """
    Spoof variant: injection perturbs ID + Data (new synthetic frame), but
    modification of an existing recorded frame keeps the ID fixed to
    target_id and perturbs only the Data field.
    """

    attack_type = 'spoof'
    default_target_id = '43f'
