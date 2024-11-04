# Presentation

[Link to the Slides](https://www.canva.com/design/DAGUpUx9tJ4/2gm42BFccb0XsUF0LnFHCg/view?utm_content=DAGUpUx9tJ4&utm_campaign=designshare&utm_medium=link&utm_source=editor)

# Notebooks

You can use [Google Colab](https://colab.research.google.com/) to open and run these notebooks. Make sure to go to Runtime and pick GPU a instance.

# Video Script

This project contains an animation script created with [Manim](https://www.manim.community/), a community-maintained mathematical animation engine.

## Prerequisites

- **Python 3.8 or higher**
- **Manim**: Install via pip if you haven't already:

  ```bash
  pip install manim
  ```

## Running the Script

To render the animation, navigate to the folder where your script is located and use the following command:

```bash
manim -pql script_name.py
```

## Replace script_name.py with the name of your file. This command will:

p: Preview the animation in a new window.
ql: Render in low quality (for quicker previews).
Additional Options

## High Quality Render:

```bash
manim -pqh script_name.py
```
