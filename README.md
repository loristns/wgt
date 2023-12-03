# WGT

**WGT** is a tiny implementation of a minimal deep learning inference runtime for the browser, through WebGPU.

Its written in TypeScript, and WGSL (WebGPU Shading Language) with the goal of being as readable as possible.

For now, it only implements the ops needed for the forward pass of [GPT-2](https://github.com/openai/gpt-2).

This is very much a work in progress, expect rough edges, unoptimized code, bugs, and missing features.

## Usage

```sh
npm install
npm run dev
```