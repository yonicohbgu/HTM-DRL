from htm.bindings.encoders import ScalarEncoderParameters

row = 9
col = 9
spars = 0.01

act_bts = 1
res = 0.1

velocity_scalar_encoder_params = ScalarEncoderParameters()
velocity_scalar_encoder_params.size = row * col
velocity_scalar_encoder_params.sparsity = spars
velocity_scalar_encoder_params.minimum = -4
velocity_scalar_encoder_params.maximum = 4
# velocity_scalar_encoder_params.resolution = res
# velocity_scalar_encoder_params.activeBits = act_bts

cart_pos_scalar_encoder_params = ScalarEncoderParameters()
cart_pos_scalar_encoder_params.size = row * col
cart_pos_scalar_encoder_params.sparsity = spars
cart_pos_scalar_encoder_params.minimum = -2.5
cart_pos_scalar_encoder_params.maximum = 2.5
# cart_pos_scalar_encoder_params.resolution = res
# cart_pos_scalar_encoder_params.activeBits = act_bts

pole_ang_scalar_encoder_params = ScalarEncoderParameters()
pole_ang_scalar_encoder_params.size = row * col
pole_ang_scalar_encoder_params.sparsity = spars
pole_ang_scalar_encoder_params.minimum = -0.3
pole_ang_scalar_encoder_params.maximum = 0.3
# pole_ang_scalar_encoder_params.resolution = res
# pole_ang_scalar_encoder_params.activeBits = act_bts

# bin_scalar_encoder_params = ScalarEncoderParameters()
# # bin_scalar_encoder_params.size = row * col
# # bin_scalar_encoder_params.sparsity = spars
# bin_scalar_encoder_params.minimum = 0
# bin_scalar_encoder_params.maximum = 1
# bin_scalar_encoder_params.resolution = 0.5
# bin_scalar_encoder_params.activeBits = int(row * col + spars)
#
# horizon_scalar_encoder_params = ScalarEncoderParameters()
# # horizon_scalar_encoder_params.size = row * col
# horizon_scalar_encoder_params.sparsity = spars
# horizon_scalar_encoder_params.minimum = 0
# horizon_scalar_encoder_params.maximum = 501

spatial_pooler_params = {
    'inputDimensions': (4 * row, col),
    'globalInhibition': True,
    'seed': 123,
    'spVerbosity': 0,
    'wrapAround': False,
    'boostStrength': 3,
    'columnDimensions': (row, col),
    'localAreaDensity': 0.04395604395604396,
    'potentialPct': 0.1,
    'synPermActiveInc': 0.06,
    'synPermConnected': 0.2,
    'synPermInactiveDec': 0.04,
    'potentialRadius': 4 * row * col
}
temporal_memory_params = {'columnDimensions': spatial_pooler_params["columnDimensions"],
                          'cellsPerColumn': 2,
                          'activationThreshold': 11,
                          'initialPermanence': 0.07,
                          'connectedPermanence': spatial_pooler_params["synPermConnected"],
                          'minThreshold': 5,
                          'maxNewSynapseCount': 32,
                          'permanenceIncrement': 0.06,
                          'permanenceDecrement': 0.03,
                          'predictedSegmentDecrement': 2 * 0.06 * 0.02,  # (roughly 2 * permInc * sparsity)
                          'maxSegmentsPerCell': 128,
                          'maxSynapsesPerSegment': 64}

# spatial_pooler_params['stimulusThreshold'] = int(round(spatial_pooler_params['stimulusThreshold']))
# spatial_pooler_params['dutyCyclePeriod'] = int(round(spatial_pooler_params['dutyCyclePeriod']))
