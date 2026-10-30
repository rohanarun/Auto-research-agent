"""Measured baseline progress and timing support are separate, policy-driven quantities."""
def assess(point, lower, policy, reference=None):
 threshold=policy['speedup_threshold']
 return {'scoring_policy_version':policy['version'],'speedup_threshold':threshold,
         'measured_speedup_exceeds_threshold':point>threshold,
         'timing_supported_above_threshold':lower>threshold,
         'historical_reference_recovered':lower>=(1-policy['equivalence_tolerance'])*reference if reference else None}
