#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

import pybullet as p



def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("-o", "--obj", help="The obj file", type=str)
	args = parser.parse_args()

	out = Path(__file__).parent/"out"
	out.mkdir(exist_ok=True)

	p.vhacd(args.obj, out/"out.obj", fileNameLogging=out/"log.txt", resolution=int(1e7), maxNumVerticesPerCH=24, depth=32,)

if __name__ == "__main__":
	main()
