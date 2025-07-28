#!/bin/bash
# Simple parser to extract author names and emails from pyproject.toml

# Files names
AUTHORS_FILE="AUTHORS"
PYPROJECT_FILE="pyproject.toml"
echo "### This is a list of pyhanami's significant contributors" > "$AUTHORS_FILE"
echo "" >> "$AUTHORS_FILE"

# Extract information between 'authors = [' and ']'
awk '/authors = \[/,/\]/' "$PYPROJECT_FILE" |
  grep '{' |
  sed -E 's/.*name = "(.*)".*email = "(.*)".*/\1 <\2>/' >> "$AUTHORS_FILE"
