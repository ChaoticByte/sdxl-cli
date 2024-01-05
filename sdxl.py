#!/usr/bin/env python3

# Copyright (c) 2024 Julian Müller (ChaoticByte)

from argparse import ArgumentParser

from diffusers import AutoPipelineForText2Image

if __name__ == "__main__":
    # parse cmdline args
    argp = ArgumentParser()
    argp.add_argument("-m", "--model", type=str, help="Path to the sdxl model folder", required=True)
    argp.add_argument("-p", "--prompt", type=str, help="Prompt for image inference")
    argp.add_argument("-n", "--steps", type=int, help="Number of inference steps", default=1)
    argp.add_argument("-o", "--output", type=str, help="Image output file", default="output.png")
    args = argp.parse_args()
    # create pipeline, process prompt(s), output file
    pipe = AutoPipelineForText2Image.from_pretrained(args.model, local_files_only=True)
    pipe.to("cpu")
    if args.prompt is None:
        try:
            while True:
                prompt = input("> ")
                pipe(prompt=prompt, num_inference_steps=args.steps, guidance_scale=0.0).images[0].save(args.output)
        except (EOFError, KeyboardInterrupt):
            print("bye.")
            exit(0)
    else:
        pipe(prompt=args.prompt, num_inference_steps=args.steps, guidance_scale=0.0).images[0].show(args.output)
