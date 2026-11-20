from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class LinearStem(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, dropout: float,
                 input_layernorm: bool = True) -> None:
        super().__init__()
        self.normalization = nn.LayerNorm(input_dim) if input_layernorm else nn.Identity()
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.linear(self.dropout(self.normalization(values))), dim=-1)


class NonlinearStem(nn.Module):
    """Two-layer, modality-specific projector with an explicit nonlinearity."""

    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int,
        input_dropout: float, hidden_dropout: float, input_layernorm: bool,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0 or not 0.0 <= hidden_dropout < 1.0:
            raise ValueError("hidden_dim must be positive and hidden_dropout in [0, 1)")
        self.normalization = nn.LayerNorm(input_dim) if input_layernorm else nn.Identity()
        self.dropout = nn.Dropout(input_dropout)
        self.hidden = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.GELU()
        self.hidden_dropout = nn.Dropout(hidden_dropout)
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        values = self.dropout(self.normalization(values))
        values = self.hidden_dropout(self.activation(self.hidden(values)))
        return F.normalize(self.output(values), dim=-1)


class ResidualStem(LinearStem):
    """A linear stem with a zero-initialized rank-32 residual in shared space."""

    def __init__(self, input_dim, output_dim, dropout, input_layernorm=True):
        super().__init__(input_dim, output_dim, dropout, input_layernorm)
        with torch.random.fork_rng():
            self.residual_down = nn.Linear(output_dim, 32, bias=False)
            self.residual_up = nn.Linear(32, output_dim, bias=False)
            nn.init.zeros_(self.residual_up.weight)

    def forward(self, values):
        z = self.linear(self.dropout(self.normalization(values)))
        return F.normalize(z + self.residual_up(F.gelu(self.residual_down(z))), dim=-1)


class DualLinearProjector(nn.Module):
    def __init__(
        self,
        face_dim: int = 512,
        voice_dim: int = 256,
        embedding_dim: int = 192,
        face_dropout: float = 0.75,
        voice_dropout: float = 0.40,
        voice_block_dropout: float = 0.0,
        voice_block_dims: tuple[int, ...] | None = None,
        input_layernorm: bool = True,
        projector_type: str = "linear",
        hidden_dim: int = 384,
        hidden_dropout: float = 0.20,
    ) -> None:
        super().__init__()
        if not 0.0 <= voice_block_dropout < 1.0:
            raise ValueError("voice_block_dropout must be in [0, 1)")
        if voice_block_dropout and (
            not voice_block_dims or len(voice_block_dims) < 2
            or any(dim <= 0 for dim in voice_block_dims)
            or sum(voice_block_dims) != voice_dim
        ):
            raise ValueError("block dropout requires valid voice_block_dims including a retained first block")
        if projector_type == "linear":
            self.face = LinearStem(face_dim, embedding_dim, face_dropout, input_layernorm)
            self.voice = LinearStem(voice_dim, embedding_dim, voice_dropout, input_layernorm)
        elif projector_type == "mlp":
            self.face = NonlinearStem(
                face_dim, hidden_dim, embedding_dim, face_dropout,
                hidden_dropout, input_layernorm,
            )
            self.voice = NonlinearStem(
                voice_dim, hidden_dim, embedding_dim, voice_dropout,
                hidden_dropout, input_layernorm,
            )
        elif projector_type == "residual":
            self.face = ResidualStem(face_dim, embedding_dim, face_dropout, input_layernorm)
            self.voice = ResidualStem(voice_dim, embedding_dim, voice_dropout, input_layernorm)
        else:
            raise ValueError(f"unknown projector_type: {projector_type}")
        self.voice_block_dropout = float(voice_block_dropout)
        self.voice_block_dims = tuple(voice_block_dims or ())

    def forward(self, faces: torch.Tensor, voices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.training and self.voice_block_dropout:
            blocks = list(torch.split(voices, self.voice_block_dims, dim=-1))
            keep_probability = 1.0 - self.voice_block_dropout
            for index in range(1, len(blocks)):
                keep = (torch.rand(voices.shape[0], 1, device=voices.device)
                        < keep_probability).to(voices.dtype)
                blocks[index] = blocks[index] * keep / keep_probability
            voices = torch.cat(blocks, dim=-1)
        return self.face(faces), self.voice(voices)


class _GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values: torch.Tensor) -> torch.Tensor:
        return values.view_as(values)

    @staticmethod
    def backward(ctx, gradient: torch.Tensor) -> torch.Tensor:
        return -gradient


class GenderAdversary(nn.Module):
    """Training-only linear gender probe with reversed projector gradient."""

    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.classifier = nn.Linear(embedding_dim, 2)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.classifier(_GradientReverse.apply(embeddings))


class ModalityStem(nn.Module):
    """Low-capacity modality-specific map before the shared identity tail."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        embedding_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Dropout(dropout),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


class SharedResidualTail(nn.Module):
    """A genuinely shared residual transform used by both modalities."""

    def __init__(self, embedding_dim: int, bottleneck_dim: int, alpha: float) -> None:
        super().__init__()
        self.normalization = nn.LayerNorm(embedding_dim)
        self.adapter = nn.Sequential(
            nn.Linear(embedding_dim, bottleneck_dim),
            nn.GELU(),
            nn.Linear(bottleneck_dim, embedding_dim),
        )
        self.alpha = float(alpha)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        residual = self.adapter(self.normalization(values))
        return F.normalize(values + self.alpha * residual, dim=-1)


class SharedIdentityPrototypeNetwork(nn.Module):
    """Separate input stems followed by one shared cross-modal identity tail."""

    def __init__(
        self,
        face_dim: int = 512,
        voice_dim: int = 256,
        hidden_dim: int = 192,
        embedding_dim: int = 96,
        shared_bottleneck_dim: int = 48,
        shared_alpha: float = 0.1,
        face_dropout: float = 0.50,
        voice_dropout: float = 0.20,
    ) -> None:
        super().__init__()
        self.face_stem = ModalityStem(
            face_dim, hidden_dim, embedding_dim, face_dropout
        )
        self.voice_stem = ModalityStem(
            voice_dim, hidden_dim, embedding_dim, voice_dropout
        )
        self.shared_tail = SharedResidualTail(
            embedding_dim, shared_bottleneck_dim, shared_alpha
        )

    def forward(
        self, faces: torch.Tensor, voices: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self.shared_tail(self.face_stem(faces)),
            self.shared_tail(self.voice_stem(voices)),
        )


def regular_simplex(classes: int, dimensions: int) -> torch.Tensor:
    """Return unit vectors with constant pairwise cosine -1/(classes-1)."""

    if classes < 2:
        raise ValueError("regular simplex requires at least two classes")
    if dimensions < classes - 1:
        raise ValueError(
            f"{classes} simplex vertices require at least {classes - 1} dimensions"
        )
    centered = torch.eye(classes) - torch.full((classes, classes), 1.0 / classes)
    _, _, right = torch.linalg.svd(centered, full_matrices=False)
    vertices = centered @ right[: classes - 1].T
    vertices = F.normalize(vertices, dim=1)
    if dimensions > classes - 1:
        vertices = F.pad(vertices, (0, dimensions - (classes - 1)))
    return vertices


class FixedSimplexClassifier(nn.Module):
    """Cosine classifier with immutable, maximally separated class targets."""

    def __init__(self, embedding_dim: int, classes: int, temperature: float) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.register_buffer("prototypes", regular_simplex(classes, embedding_dim))
        self.temperature = float(temperature)

    def logits(self, embeddings: torch.Tensor) -> torch.Tensor:
        return F.linear(F.normalize(embeddings), self.prototypes) / self.temperature

    def loss(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(self.logits(embeddings), labels)


def cross_modal_supervised_contrastive_loss(
    face_embeddings: torch.Tensor,
    voice_embeddings: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Symmetric multi-positive InfoNCE using only face-to-voice comparisons."""

    if temperature <= 0:
        raise ValueError("temperature must be positive")
    face = F.normalize(face_embeddings, dim=1)
    voice = F.normalize(voice_embeddings, dim=1)
    positive = labels[:, None].eq(labels[None, :]).to(face.dtype)
    if torch.any(positive.sum(dim=1) == 0):
        raise ValueError("each anchor needs a cross-modal positive")

    def direction(logits: torch.Tensor) -> torch.Tensor:
        logits = logits / temperature
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()
        log_probability = logits - torch.logsumexp(logits, dim=1, keepdim=True)
        return -((positive * log_probability).sum(dim=1) / positive.sum(dim=1)).mean()

    similarities = face @ voice.T
    return 0.5 * (direction(similarities) + direction(similarities.T))


def intra_modal_supervised_contrastive_loss(
    face_embeddings: torch.Tensor,
    voice_embeddings: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Average face/face and voice/voice SupCon, excluding each anchor itself."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if face_embeddings.shape != voice_embeddings.shape or face_embeddings.ndim != 2:
        raise ValueError("face/voice embeddings must be matching matrices")
    if labels.ndim != 1 or labels.shape[0] != face_embeddings.shape[0]:
        raise ValueError("labels must match batch rows")
    identity = torch.eye(labels.numel(), device=labels.device, dtype=torch.bool)
    positive = labels[:, None].eq(labels[None, :]) & ~identity
    if torch.any(positive.sum(dim=1) == 0):
        raise ValueError("each anchor needs another sample of its identity")

    def modality(embeddings: torch.Tensor) -> torch.Tensor:
        normalized = F.normalize(embeddings, dim=1)
        logits = (normalized @ normalized.T) / temperature
        logits = logits.masked_fill(identity, -torch.inf)
        log_probability = logits - torch.logsumexp(logits, dim=1, keepdim=True)
        return -(
            log_probability.masked_fill(~positive, 0.0).sum(dim=1)
            / positive.sum(dim=1)
        ).mean()

    return 0.5 * (modality(face_embeddings) + modality(voice_embeddings))


def gender_weighted_cross_modal_contrastive_loss(
    face_embeddings: torch.Tensor,
    voice_embeddings: torch.Tensor,
    labels: torch.Tensor,
    genders: torch.Tensor,
    temperature: float,
    same_gender_negative_weight: float,
) -> torch.Tensor:
    """Multi-positive SupCon with a soft weight on same-gender impostors.

    A weight of one is exactly the ordinary cross-modal SupCon objective.
    All positives retain unit weight; the weight affects only denominator
    terms for different identities sharing the anchor's gender.
    """
    if temperature <= 0 or same_gender_negative_weight < 1:
        raise ValueError("temperature must be positive and negative weight >= 1")
    face = F.normalize(face_embeddings, dim=1)
    voice = F.normalize(voice_embeddings, dim=1)
    positive = labels[:, None].eq(labels[None, :]).to(face.dtype)
    negative_same_gender = genders[:, None].eq(genders[None, :]) & ~positive.bool()
    log_weight = negative_same_gender.to(face.dtype) * math.log(same_gender_negative_weight)

    def direction(logits: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        logits = logits / temperature
        centered = logits - logits.max(dim=1, keepdim=True).values.detach()
        log_probability = centered - torch.logsumexp(centered + weights, dim=1, keepdim=True)
        return -((positive * log_probability).sum(dim=1) / positive.sum(dim=1)).mean()

    similarities = face @ voice.T
    return 0.5 * (
        direction(similarities, log_weight)
        + direction(similarities.T, log_weight.T)
    )


def same_gender_cross_modal_contrastive_loss(
    face_embeddings: torch.Tensor,
    voice_embeddings: torch.Tensor,
    labels: torch.Tensor,
    genders: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Cross-modal SupCon with only training-set same-gender candidates.

    This is an auxiliary training loss: inference needs neither gender labels
    nor other trial pairs. The regular unconditioned SupCon is kept separately.
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if labels.ndim != 1 or genders.ndim != 1 or labels.shape != genders.shape:
        raise ValueError("labels and genders must match batch rows")
    face = F.normalize(face_embeddings, dim=1)
    voice = F.normalize(voice_embeddings, dim=1)
    allowed = genders[:, None].eq(genders[None, :])
    positive = labels[:, None].eq(labels[None, :]) & allowed
    if torch.any(positive.sum(dim=1) == 0):
        raise ValueError("each anchor needs a same-gender cross-modal positive")

    def direction(logits: torch.Tensor) -> torch.Tensor:
        logits = (logits / temperature).masked_fill(~allowed, -torch.inf)
        log_probability = logits - torch.logsumexp(logits, dim=1, keepdim=True)
        return -(
            log_probability.masked_fill(~positive, 0.0).sum(dim=1)
            / positive.sum(dim=1)
        ).mean()

    similarities = face @ voice.T
    return 0.5 * (direction(similarities) + direction(similarities.T))


def paired_alignment_loss(
    face_embeddings: torch.Tensor, voice_embeddings: torch.Tensor
) -> torch.Tensor:
    return (1.0 - F.cosine_similarity(face_embeddings, voice_embeddings, dim=1)).mean()


def batch_identity_prototype_alignment_loss(
    face_embeddings: torch.Tensor,
    voice_embeddings: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """P4c-style squared distance between modality centroids per batch identity."""
    if face_embeddings.shape != voice_embeddings.shape or face_embeddings.ndim != 2:
        raise ValueError("face/voice embeddings must be matching matrices")
    if labels.ndim != 1 or labels.shape[0] != face_embeddings.shape[0]:
        raise ValueError("labels must match batch rows")
    if labels.numel() == 0:
        raise ValueError("empty batch")
    _, inverse = torch.unique(labels, sorted=True, return_inverse=True)
    identities = int(inverse.max().item()) + 1
    dimension = face_embeddings.shape[1]
    face_sum = face_embeddings.new_zeros((identities, dimension))
    voice_sum = voice_embeddings.new_zeros((identities, dimension))
    face_sum.index_add_(0, inverse, face_embeddings)
    voice_sum.index_add_(0, inverse, voice_embeddings)
    count = torch.bincount(inverse, minlength=identities).to(face_embeddings.dtype)
    return ((face_sum / count[:, None]) - (voice_sum / count[:, None])).square().sum(dim=1).mean()


def fixed_simplex_center_alignment_loss(
    face_embeddings: torch.Tensor,
    voice_embeddings: torch.Tensor,
    labels: torch.Tensor,
    centers: torch.Tensor,
) -> torch.Tensor:
    """Attract both modalities to immutable train-identity simplex vertices."""
    if face_embeddings.shape != voice_embeddings.shape or face_embeddings.ndim != 2:
        raise ValueError("face and voice embedding shapes must match")
    if labels.shape != (face_embeddings.shape[0],):
        raise ValueError("labels must match batch rows")
    if centers.ndim != 2 or centers.shape[1] != face_embeddings.shape[1]:
        raise ValueError("center dimension differs from embedding dimension")
    if torch.any(labels < 0) or torch.any(labels >= len(centers)):
        raise ValueError("label outside fixed centers")
    target = centers[labels]
    return 0.5 * (
        (face_embeddings - target).square().sum(dim=1).mean()
        + (voice_embeddings - target).square().sum(dim=1).mean()
    )


def same_gender_hard_negative_loss(
    face_embeddings: torch.Tensor,
    voice_embeddings: torch.Tensor,
    labels: torch.Tensor,
    genders: torch.Tensor,
    margin: float = 0.2,
) -> torch.Tensor:
    """Bidirectional hardest-negative ranking loss within the same gender."""

    face = F.normalize(face_embeddings, dim=1)
    voice = F.normalize(voice_embeddings, dim=1)
    similarity = face @ voice.T
    same_identity = labels[:, None].eq(labels[None, :])
    valid_negative = genders[:, None].eq(genders[None, :]) & ~same_identity
    positive = (similarity * same_identity).sum(dim=1) / same_identity.sum(dim=1).clamp_min(1)

    def direction(scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        hard = scores.masked_fill(~mask, -torch.inf).max(dim=1).values
        valid = torch.isfinite(hard)
        if not torch.any(valid):
            return scores.new_zeros(())
        return F.relu(float(margin) + hard[valid] - positive[valid]).mean()

    return 0.5 * (
        direction(similarity, valid_negative)
        + direction(similarity.T, valid_negative.T)
    )


def age_matched_same_gender_hard_negative_loss(
    face_embeddings: torch.Tensor,
    voice_embeddings: torch.Tensor,
    labels: torch.Tensor,
    genders: torch.Tensor,
    face_ages: torch.Tensor,
    voice_ages: torch.Tensor,
    nearest_k: int = 8,
    margin: float = 0.20,
    temperature: float = 0.10,
) -> torch.Tensor:
    """Smooth bidirectional hardest negative among age-near same-gender IDs.

    Age/gender are training metadata only.  For each anchor, rank eligible
    impostors by predicted-age proximity, then mine the highest *current*
    cross-modal cosine among the nearest K.  No evaluation-trial context is
    used by this loss or by the resulting projector at inference.
    """
    if nearest_k < 1 or temperature <= 0:
        raise ValueError("nearest_k and temperature must be positive")
    if face_embeddings.shape != voice_embeddings.shape or face_embeddings.ndim != 2:
        raise ValueError("face and voice embedding shapes must match")
    count = face_embeddings.shape[0]
    if any(t.shape != (count,) for t in (labels, genders, face_ages, voice_ages)):
        raise ValueError("labels, genders and ages must match batch rows")
    face = F.normalize(face_embeddings, dim=1)
    voice = F.normalize(voice_embeddings, dim=1)
    similarity = face @ voice.T
    positive_mask = labels[:, None].eq(labels[None, :])
    negative_mask = genders[:, None].eq(genders[None, :]) & ~positive_mask
    age_distance = (face_ages[:, None] - voice_ages[None, :]).abs()

    def direction(scores: torch.Tensor, positives: torch.Tensor,
                  negatives: torch.Tensor, ages: torch.Tensor) -> torch.Tensor:
        nearest = ages.masked_fill(~negatives, torch.inf)
        k = min(nearest_k, nearest.shape[1])
        indices = nearest.topk(k, dim=1, largest=False).indices
        selected = torch.zeros_like(negatives)
        selected.scatter_(1, indices, True)
        selected &= negatives
        hard = scores.masked_fill(~selected, -torch.inf).max(dim=1).values
        valid = torch.isfinite(hard)
        if not torch.any(valid):
            return scores.new_zeros(())
        positive = (scores * positives).sum(dim=1) / positives.sum(dim=1).clamp_min(1)
        violation = hard[valid] - positive[valid] + float(margin)
        return (float(temperature) * F.softplus(violation / float(temperature))).mean()

    return 0.5 * (
        direction(similarity, positive_mask, negative_mask, age_distance)
        + direction(similarity.T, positive_mask.T, negative_mask.T, age_distance.T)
    )


def curriculum_cross_modal_negative_loss(
    face_embeddings: torch.Tensor,
    voice_embeddings: torch.Tensor,
    labels: torch.Tensor,
    genders: torch.Tensor,
    face_ages: torch.Tensor,
    voice_ages: torch.Tensor,
    epoch: int,
    nearest_k: int = 8,
    margin: float = 0.20,
    temperature: float = 0.10,
) -> torch.Tensor:
    """Train-only random → same-gender → age-near semi-hard curriculum.

    Both directions mine exclusively within the current fit batch.  Predicted
    ages and gender are supplied only for fold-training rows by the caller.
    """
    if epoch < 1 or nearest_k < 1 or temperature <= 0:
        raise ValueError("epoch, nearest_k and temperature must be positive")
    if face_embeddings.shape != voice_embeddings.shape or face_embeddings.ndim != 2:
        raise ValueError("face and voice embeddings must be matching matrices")
    count = len(labels)
    if any(values.shape != (count,) for values in (labels, genders, face_ages, voice_ages)):
        raise ValueError("labels, genders and ages must match batch rows")
    similarity = F.normalize(face_embeddings, dim=1) @ F.normalize(voice_embeddings, dim=1).T
    positive_mask = labels[:, None].eq(labels[None, :])
    negative_mask = ~positive_mask
    same_gender = genders[:, None].eq(genders[None, :]) & negative_mask
    age_distance = (face_ages[:, None] - voice_ages[None, :]).abs()
    stage = 0 if epoch <= 3 else (1 if epoch <= 8 else 2)
    weights = ((0.5, 0.5, 0.0), (0.2, 0.4, 0.4), (0.1, 0.2, 0.7))[stage]

    def direction(scores: torch.Tensor, positives: torch.Tensor,
                  negatives: torch.Tensor, same: torch.Tensor,
                  distances: torch.Tensor) -> torch.Tensor:
        positive = (scores * positives).sum(dim=1) / positives.sum(dim=1).clamp_min(1)

        def sampled(mask: torch.Tensor) -> torch.Tensor:
            valid = mask.any(dim=1)
            if not valid.any():
                return scores.new_zeros(())
            indices = torch.multinomial(mask[valid].float(), 1).squeeze(1)
            chosen = scores[valid].gather(1, indices[:, None]).squeeze(1)
            violation = (chosen - positive[valid] + margin) / temperature
            return (temperature * F.softplus(violation)).mean()

        random_loss = sampled(negatives)
        same_loss = sampled(same)
        if weights[2] == 0:
            return weights[0] * random_loss + weights[1] * same_loss
        nearest = distances.masked_fill(~same, torch.inf)
        indices = nearest.topk(min(nearest_k, count), dim=1, largest=False).indices
        selected = torch.zeros_like(same)
        selected.scatter_(1, indices, True)
        selected &= same
        hard = scores.masked_fill(~selected, -torch.inf).max(dim=1).values
        if stage == 2:
            # Prefer an informative impostor below the positive score;
            # fall back to the hardest age-near impostor if none exists.
            semihard = selected & scores.lt(positive[:, None])
            semi = scores.masked_fill(~semihard, -torch.inf).max(dim=1).values
            hard = torch.where(torch.isfinite(semi), semi, hard)
        valid = torch.isfinite(hard)
        if valid.any():
            violation = (hard[valid] - positive[valid] + margin) / temperature
            age_loss = (temperature * F.softplus(violation)).mean()
        else:
            age_loss = scores.new_zeros(())
        return (weights[0] * random_loss + weights[1] * same_loss
                + weights[2] * age_loss)

    return 0.5 * (
        direction(similarity, positive_mask, negative_mask, same_gender, age_distance)
        + direction(similarity.T, positive_mask.T, negative_mask.T,
                    same_gender.T, age_distance.T)
    )


def covariance_decorrelation_loss(embeddings: torch.Tensor) -> torch.Tensor:
    """Penalize redundant feature correlations without forcing unit covariance."""

    if embeddings.shape[0] < 2:
        return embeddings.new_zeros(())
    centered = embeddings - embeddings.mean(dim=0, keepdim=True)
    standardized = centered / centered.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-4)
    correlation = standardized.T @ standardized / embeddings.shape[0]
    off_diagonal = correlation - torch.diag_embed(torch.diagonal(correlation))
    return off_diagonal.square().mean()


class SharedAAMClassifier(nn.Module):
    def __init__(self, embedding_dim: int, classes: int, margin: float, scale: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)
        self.margin = float(margin)
        self.scale = float(scale)
        self.cos_m = math.cos(self.margin)
        self.sin_m = math.sin(self.margin)
        self.threshold = math.cos(math.pi - self.margin)
        self.correction = math.sin(math.pi - self.margin) * self.margin

    def logits(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        cosine = F.linear(F.normalize(embeddings), F.normalize(self.weight)).clamp(-1 + 1e-7, 1 - 1e-7)
        sine = torch.sqrt(torch.clamp(1.0 - cosine.square(), min=0.0))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.threshold, phi, cosine - self.correction)
        one_hot = F.one_hot(labels, num_classes=self.weight.shape[0]).to(cosine.dtype)
        return self.scale * (one_hot * phi + (1.0 - one_hot) * cosine)

    def loss(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(self.logits(embeddings, labels), labels)

    def same_gender_loss(
        self, embeddings: torch.Tensor, labels: torch.Tensor, class_genders: torch.Tensor,
    ) -> torch.Tensor:
        """Training-only AAM competition among same-gender identity centers.

        Reuses the ordinary identity head; adds no parameters or inference gate.
        Genders describe fitted classifier classes, never evaluation examples.
        """
        if class_genders.ndim != 1 or len(class_genders) != len(self.weight):
            raise ValueError("one gender code is required per fitted identity class")
        class_genders = class_genders.to(labels.device)
        if not torch.all((class_genders == 0) | (class_genders == 1)):
            raise ValueError("classifier gender codes must be binary")
        allowed = class_genders[labels, None].eq(class_genders[None, :])
        logits = self.logits(embeddings, labels).masked_fill(~allowed, float("-inf"))
        return F.cross_entropy(logits, labels)
