import torch

from .attack_utilities import calculate_crc, stuff_bits



def compute_row_gradient_magnitude(data_grad, row_idx):
    return data_grad[:, :, row_idx, :].abs().sum(dim=(1, 2))


def update_max_grad(row_grad_magnitude, max_grad, max_grad_row, row_idx, all_green):
    update_mask = (row_grad_magnitude > max_grad) & all_green
    max_grad = torch.where(update_mask, row_grad_magnitude, max_grad)
    max_grad_row = torch.where(update_mask, torch.tensor(row_idx, device=max_grad.device), max_grad_row)
    return max_grad, max_grad_row


def create_mask_for_max_grad_row(mask, max_grad_row, image_shape):
    for b in range(image_shape[0]):
        mask[b, :, max_grad_row[b], :] = 1
    return mask


def initialize_max_grad_variables(batch_size, num_rows, device):
    max_grad = torch.zeros(batch_size, device=device)
    max_grad_row = torch.zeros(batch_size, dtype=torch.long, device=device)
    return max_grad, max_grad_row


def extract_color_channels(image):
    return image[:, 0, :, :], image[:, 1, :, :], image[:, 2, :, :]


def create_green_mask(red_channel, green_channel, blue_channel):
    return (red_channel == 0) & (green_channel == 1) & (blue_channel == 0)


def initialize_mask(image):
    return torch.zeros_like(image, dtype=torch.float)


def generate_max_grad_mask(image, data_grad):
    red_channel, green_channel, blue_channel = extract_color_channels(image)
    green_mask = create_green_mask(red_channel, green_channel, blue_channel)
    max_grad, max_grad_row = initialize_max_grad_variables(green_channel.shape[0], green_channel.shape[1], image.device)

    updated_flag = False

    for i in range(green_channel.shape[1]):
        all_green = green_mask[:, i, :].all(dim=1)
        row_grad_magnitude = compute_row_gradient_magnitude(data_grad, i)

        prev_max_grad = max_grad.clone()
        max_grad, max_grad_row = update_max_grad(row_grad_magnitude, max_grad, max_grad_row, i, all_green)

        if not torch.equal(prev_max_grad, max_grad):
            updated_flag = True

    mask = initialize_mask(data_grad)
    print("max_grad_row_indices for injection: ", max_grad_row.item())
    mask = create_mask_for_max_grad_row(mask, max_grad_row, image.shape)

    if not updated_flag:
        return None

    return mask



def find_max_perturbations(image, pattern_length, rgb_pattern, matched_rows, ifprint):
    if matched_rows is None:
        matched_rows = []

    for i in range(image.shape[2]):
        matches_pattern = torch.ones(image.shape[0], dtype=torch.bool, device=image.device)

        for j in range(pattern_length):
            r, g, b = rgb_pattern[j]
            matches_pattern &= (image[:, 0, i, j] == r) & (image[:, 1, i, j] == g) & (image[:, 2, i, j] == b)

        if matches_pattern.any():
            matched_rows.extend([i for b in range(image.shape[0]) if matches_pattern[b]])

    if ifprint:
        print("Initial matched rows for modification:", matched_rows)

    return matched_rows, len(matched_rows)


def select_row_to_perturb(mask, data_grad, matched_rows, selected_rows_set):
    gradients = []

    for row in matched_rows:
        if row in selected_rows_set:
            continue

        row_mask = mask[:, :, row, :].bool()
        row_grad = data_grad[:, :, row, :]

        gradient_magnitude = row_grad.abs() * row_mask
        gradients.append((row, gradient_magnitude.sum().item()))

    if gradients:
        selected_row, _ = max(gradients, key=lambda x: x[1])

        updated_mask = torch.zeros_like(mask)
        updated_mask[:, :, selected_row, :] = mask[:, :, selected_row, :]

        selected_rows_set.add(selected_row)
        return selected_row, updated_mask, selected_rows_set

    return None, torch.zeros_like(mask), selected_rows_set


def generate_mask_modify(image, data_grad, matched_rows, selected_rows_set, bit_pattern, perturb_id):
    """
    perturb_id=True  -> mask covers both the ID field and the Data field (DoS).
    perturb_id=False -> mask covers only the Data field, ID field is left
                         untouched by the FGSM step (Spoof modification).
    """
    sof_len = 1
    id_mask_length = 11
    mid_bits_length = 7

    if selected_rows_set is None:
        selected_rows_set = set()

    mask = torch.zeros_like(data_grad)

    rgb_pattern = [(0.0, 0.0, 0.0) if bit == '0' else (1.0, 1.0, 1.0) for bit in bit_pattern]
    pattern_length = len(rgb_pattern)

    if not matched_rows:
        matched_rows, _ = find_max_perturbations(image, pattern_length, rgb_pattern, matched_rows, ifprint=True)

    filtered_matched_rows = [row for row in matched_rows if row not in selected_rows_set]

    if not filtered_matched_rows:
        print("[WARN] No rows left to perturb for this image.")
        return torch.zeros_like(mask), matched_rows, selected_rows_set

    for row in filtered_matched_rows:
        for b in range(image.shape[0]):
            if perturb_id:
                mask[b, :, row, sof_len:sof_len + id_mask_length] = 1
            mask[b, :, row, sof_len + id_mask_length + mid_bits_length: sof_len + id_mask_length + mid_bits_length + 64] = 1

    selected_row, updated_mask, selected_rows_set = select_row_to_perturb(mask, data_grad, filtered_matched_rows, selected_rows_set)
    print("selected row for modification: ", selected_row)
    selected_rows_set.add(selected_row)

    return updated_mask, matched_rows, selected_rows_set




def _decode_bits(pixel_source, b, row, col_start, col_end, device):
    bits = ''
    for col in range(col_start, col_end):
        pix = pixel_source[b, :, row, col]
        dot1 = torch.dot(pix, torch.tensor([1.0, 1.0, 1.0], device=device))
        dot0 = torch.dot(pix, torch.tensor([0.0, 0.0, 0.0], device=device))
        bits += '1' if dot1 >= dot0 else '0'
    return bits


def gradient_perturbation_targeted(image, perturbed_image, mask, mode, target_id_bits=None):
    """
    mode='id_data'   -> ID and Data are both taken from the FGSM-perturbed
                         pixels as-is (no projection to an existing ID).
    mode='data_only' -> ID is fixed to target_id_bits (kept as-is); only
                         Data is taken from the FGSM-perturbed pixels.
    """
    ID_len = 11
    middle_bits = "0001000"

    if mode == 'data_only' and target_id_bits is None:
        raise ValueError("target_id_bits is required when mode='data_only'")

    for b in range(image.shape[0]):
        rows = mask[b, 0].nonzero(as_tuple=True)[0]
        rows = torch.unique(rows)

        for row in rows:
            if mode == 'id_data':
                id_bits = _decode_bits(perturbed_image, b, row, 1, 1 + ID_len, image.device)
            else:
                id_bits = target_id_bits

            data_start = 1 + ID_len + len(middle_bits)
            data_bits = _decode_bits(perturbed_image, b, row, data_start, data_start + 64, image.device)

            frame_start = '0' + id_bits + middle_bits + data_bits
            crc_bits = bin(calculate_crc(frame_start))[2:].zfill(15)
            stuffed = stuff_bits(frame_start + crc_bits)

            for i, bit in enumerate(stuffed):
                perturbed_image[b, :, row, i] = 1.0 if bit == '1' else 0.0

            ending = '1011111111111'
            offset = len(stuffed)
            for i, bit in enumerate(ending):
                perturbed_image[b, :, row, offset + i] = 1.0 if bit == '1' else 0.0

            for i in range(offset + len(ending), perturbed_image.shape[-1]):
                perturbed_image[b, 1, row, i] = 1.0
                perturbed_image[b, 0, row, i] = 0.0
                perturbed_image[b, 2, row, i] = 0.0

    return perturbed_image
