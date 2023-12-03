import {Tensor} from '../../wgt/tensor';
import {DeviceTensor} from '../../wgt/deviceTensor';

import {parameters, Parameters} from '../../wgt/ops/parameters';
import {merge, MergeMethod} from '../../wgt/ops/merge';
import {layerNorm, LayerNormParameters} from '../../wgt/ops/layerNorm';

import {gpt2Block, Gpt2BlockParameters} from './gpt2Block';
import {embed, EmbedParameters} from './embed';
import {deEmbed} from './deEmbed';

export interface Gpt2Parameters extends Parameters {
  tokenEmbeddings: EmbedParameters;
  positionEmbeddings: EmbedParameters;
  blocks: Gpt2BlockParameters[];
  layerNorm: LayerNormParameters;
}

export function gpt2(
  tokens: DeviceTensor,
  params: Gpt2Parameters
): DeviceTensor {
  const {positions} = parameters({
    positions: Tensor.fromArray([...Array(tokens.shape.cols).keys()]),
  });

  const tokenEmbeddings = embed(tokens, params.tokenEmbeddings);
  const positionEmbeddings = embed(positions, params.positionEmbeddings);

  let x = merge(tokenEmbeddings, positionEmbeddings, MergeMethod.Add);

  for (let i = 0; i < params.blocks.length; i++) {
    x = gpt2Block(x, params.blocks[i]);
  }

  x = layerNorm(x, params.layerNorm);
  x = deEmbed(x, params.tokenEmbeddings);

  return x;
}
