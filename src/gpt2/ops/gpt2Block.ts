import {DeviceTensor} from '../../wgt/deviceTensor';

import {Parameters} from '../../wgt/ops/parameters';
import {
  selfAttention,
  SelfAttentionParameters,
} from '../../wgt/ops/selfAttention';
import {layerNorm, LayerNormParameters} from '../../wgt/ops/layerNorm';
import {merge, MergeMethod} from '../../wgt/ops/merge';

import {feedForward, FeedForwardParameters} from './feedForward';

export interface Gpt2BlockParameters extends Parameters {
  layerNorm1: LayerNormParameters;
  selfAttention: SelfAttentionParameters;
  layerNorm2: LayerNormParameters;
  feedForward: FeedForwardParameters;
}

/**
 * A GPT-2 block (with a self-attention layer and a feed-forward layer)
 */
export function gpt2Block(
  input: DeviceTensor,
  params: Gpt2BlockParameters
): DeviceTensor {
  const ln1 = layerNorm(input, params.layerNorm1);
  const att = selfAttention(ln1, params.selfAttention);
  const res1 = merge(input, att, MergeMethod.Add);

  const ln2 = layerNorm(res1, params.layerNorm2);
  const ffn = feedForward(ln2, params.feedForward);
  const res2 = merge(res1, ffn, MergeMethod.Add);

  return res2;
}
