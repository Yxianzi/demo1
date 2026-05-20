import torch


class DistributionCalibrator:
    def __init__(
        self,
        num_classes,
        momentum=0.9,
        tau=0.3,
        eps=1e-6,
        device="cuda",
    ):
        self.num_classes = num_classes
        self.momentum = momentum
        self.tau = tau
        self.eps = eps
        self.device = torch.device(device)
        self.prior = torch.ones(num_classes, device=self.device) / num_classes

    def update(self, prob):
        prob = prob.detach().to(self.prior.device)
        batch_prior = prob.mean(dim=0)
        self.prior = self.momentum * self.prior + (1.0 - self.momentum) * batch_prior
        self.prior = self.prior / self.prior.sum().clamp_min(self.eps)

    def calibrate(self, prob):
        prob = prob.detach().to(self.prior.device)
        adjusted = prob / self.prior.clamp_min(self.eps).pow(self.tau).unsqueeze(0)
        adjusted = adjusted / adjusted.sum(dim=1, keepdim=True).clamp_min(self.eps)
        return adjusted

    def update_and_calibrate(self, prob):
        self.update(prob)
        return self.calibrate(prob)


def blend_target_probability(student_prob, calibrated_prob, rho):
    return (1.0 - rho) * student_prob + rho * calibrated_prob
