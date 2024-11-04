from manim import *
import numpy as np

class TextAnimation(Scene):
    def construct(self):
        # Initialize text
        text = "Early one morning the sun was shining I was laying in bed Wondering if she had changed at all if her hair was still red"
        words = text.split()

        # Calculate positions in a snake-like pattern
        positions = []
        rows = 4  # Number of rows
        cols = max(len(words) // rows + 1, 6)  # Minimum 6 columns
        
        # Calculate spacing
        x_spacing = config.frame_width * 0.8 / (cols - 1)  # Leave 10% margin on each side
        y_spacing = config.frame_height * 0.6 / (rows - 1)  # Leave 20% margin on top/bottom
        
        # Starting position (top-left)
        start_x = -config.frame_width * 0.4  # Start 40% from left edge
        start_y = config.frame_height * 0.3   # Start 30% from top
        
        # Create positions in a snake pattern
        current_word = 0
        for row in range(rows):
            for col in range(cols):
                if current_word >= len(words):
                    break
                    
                # Calculate x position (alternate direction for each row)
                if row % 2 == 0:
                    x = start_x + (col * x_spacing)
                else:
                    x = start_x + ((cols - 1 - col) * x_spacing)
                    
                y = start_y - (row * y_spacing)
                
                positions.append(np.array([x, y, 0]))
                current_word += 1
                if current_word >= len(words):
                    break

        # Create text objects and nodes
        text_objects = []
        nodes = []
        for i, word in enumerate(words):
            text_obj = Text(word, font_size=24)
            node = Circle(radius=0.3, color=WHITE)
            
            # Position both text and node
            node.move_to(positions[i])
            text_obj.move_to(positions[i])
            
            text_objects.append(text_obj)
            nodes.append(node)

        # Create arrows between nodes
        arrows = []
        for i in range(len(nodes) - 1):
            start_node = nodes[i]
            end_node = nodes[i + 1]
            
            # Calculate arrow direction vector
            direction = end_node.get_center() - start_node.get_center()
            unit_vector = direction / np.linalg.norm(direction)
            
            # Create arrow with proper positioning
            arrow = Arrow(
                start_node.get_center() + unit_vector * 0.3,
                end_node.get_center() - unit_vector * 0.3,
                buff=0,
                max_tip_length_to_length_ratio=0.15,
                stroke_width=2  # Thinner arrows
            )
            arrows.append(arrow)

        # Initial animation
        self.play(
            *[Create(node) for node in nodes],
            *[Write(text_obj) for text_obj in text_objects],
            run_time=1
        )
        self.play(*[Create(arrow) for arrow in arrows], run_time=1)
        self.wait()

        # Identify and color 'was' nodes
        was_indices = [i for i, word in enumerate(words) if word.lower() == "was"]
        was_nodes = [nodes[i] for i in was_indices]
        was_texts = [text_objects[i] for i in was_indices]

        self.play(
            *[node.animate.set_color(RED) for node in was_nodes],
            *[text.animate.set_color(RED) for text in was_texts],
            run_time=1
        )
        self.wait()

        # Calculate merged node position (center of mass of was nodes)
        center_pos = np.mean([node.get_center() for node in was_nodes], axis=0)
        merged_node = Circle(radius=0.5, color=RED)
        merged_text = Text("was", font_size=24, color=RED)
        merged_node.move_to(center_pos)
        merged_text.move_to(center_pos)

        # Prepare arrow animations
        new_arrows = []
        arrows_to_remove = []

        # For each was node, update connected arrows
        for idx in was_indices:
            if idx > 0:
                arrows_to_remove.append(arrows[idx - 1])
                prev_node = nodes[idx - 1]
                new_arrow = Arrow(
                    prev_node.get_center(),
                    merged_node.get_center(),
                    buff=0.3,
                    max_tip_length_to_length_ratio=0.15,
                    stroke_width=2
                )
                new_arrows.append(new_arrow)

            if idx < len(words) - 1:
                arrows_to_remove.append(arrows[idx])
                next_node = nodes[idx + 1]
                new_arrow = Arrow(
                    merged_node.get_center(),
                    next_node.get_center(),
                    buff=0.3,
                    max_tip_length_to_length_ratio=0.15,
                    stroke_width=2
                )
                new_arrows.append(new_arrow)

        # Animate merging
        self.play(
            *[node.animate.move_to(center_pos) for node in was_nodes],
            *[text.animate.move_to(center_pos) for text in was_texts],
            *[FadeOut(arrow) for arrow in arrows_to_remove],
            run_time=1
        )

        self.play(
            ReplacementTransform(VGroup(*was_nodes), merged_node),
            ReplacementTransform(VGroup(*was_texts), merged_text),
            *[Create(arrow) for arrow in new_arrows],
            run_time=1
        )

        self.wait(2)

if __name__ == "__main__":
    scene = TextAnimation()
    scene.render()
