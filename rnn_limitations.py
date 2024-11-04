import numpy as np
from manim import *


class RNNLimitationsAnimation(Scene):
    def create_token_box(self, text, color=WHITE):
        token = Text(text, font_size=24)
        box = SurroundingRectangle(token, color=color, corner_radius=0.1)
        return VGroup(token, box)

    def create_hidden_state(self, label="h", color=BLUE):
        circle = Circle(radius=0.3, color=color)
        text = MathTex(label, color=color)
        text.move_to(circle.get_center())
        return VGroup(circle, text)

    def create_timeline_marker(self, time, color=GRAY):
        line = Line(UP * 0.2, DOWN * 0.2, color=color)
        text = Text(f"t={time}", font_size=16, color=color)
        text.next_to(line, DOWN, buff=0.1)
        return VGroup(line, text)

    def construct(self):
        # Colors
        RNN_COLOR = "#2ecc71"
        TRANSFORMER_COLOR = "#3498db"
        HIDDEN_COLOR = "#e74c3c"
        ATTENTION_COLOR = "#f1c40f"

        # Title and introduction
        title = Text("Understanding the Limitations of RNNs", font_size=40)
        title.to_edge(UP, buff=0.5)

        self.play(Write(title))

        # RNN Section
        rnn_title = Text(
            "Recurrent Neural Network (RNN)", color=RNN_COLOR, font_size=32
        )
        rnn_title.next_to(title, DOWN, buff=1.5)

        # Create input sequence
        input_sequence = ["ICTer", "workshop", "2024", "is", "awesome"]
        token_boxes = VGroup(*[self.create_token_box(word) for word in input_sequence])
        token_boxes.arrange(RIGHT, buff=0.75)
        token_boxes.next_to(
            rnn_title, DOWN, buff=1.5
        )  # Increased buff to avoid overlap

        self.play(Write(rnn_title))
        self.play(Create(token_boxes))

        # RNN processing visualization
        hidden_states = VGroup()
        arrows = VGroup()
        time_markers = VGroup()

        # Initial hidden state
        h0 = self.create_hidden_state("h_0", HIDDEN_COLOR)
        h0.next_to(token_boxes[0], UP, buff=0.75)
        hidden_states.add(h0)

        # Show initial setup
        self.play(Create(h0))

        # Process each token sequentially
        for i in range(len(input_sequence)):
            # Create new hidden state
            hi = self.create_hidden_state(f"h_{i+1}", HIDDEN_COLOR)
            hi.next_to(token_boxes[i], UP, buff=0.75)
            # Create arrows
            input_arrow = Arrow(
                token_boxes[i].get_top(), hi.get_bottom(), color=RNN_COLOR
            )
            if i > 0:
                state_arrow = Arrow(
                    hidden_states[-1].get_right(), hi.get_left(), color=HIDDEN_COLOR
                )
                arrows.add(state_arrow)

            arrows.add(input_arrow)
            hidden_states.add(hi)

            # Time marker
            marker = self.create_timeline_marker(i + 1)
            marker.next_to(token_boxes[i], DOWN)
            time_markers.add(marker)

            # Animate sequential processing
            self.play(
                Create(hi),
                GrowArrow(input_arrow),
                Create(marker),
                token_boxes[i].animate.set_color(RNN_COLOR),
                run_time=0.8,
            )
            if i > 0:
                self.play(GrowArrow(state_arrow), run_time=0.5)

            # Add processing time indicator
            wait_text = Text("Processing...", font_size=16, color=GRAY)
            wait_text.next_to(hi, RIGHT)
            self.play(FadeIn(wait_text))
            self.play(FadeOut(wait_text))

        # Hide token boxes and title before showing limitations
        self.play(
            FadeOut(token_boxes),
            FadeOut(rnn_title),
            FadeOut(h0),
            *[FadeOut(hi) for hi in hidden_states],
            *[FadeOut(marker) for marker in time_markers],
            *[FadeOut(input_arrow) for input_arrow in arrows],
            *[FadeOut(state_arrow) for state_arrow in arrows if state_arrow in arrows],
        )

        # Transformer Comparison
        transformer_title = Text(
            "Transformer (Parallel Processing)", color=TRANSFORMER_COLOR, font_size=32
        )
        transformer_title.next_to(title, RIGHT, buff=2)

        # # Create parallel attention visualization
        # attention_lines = VGroup()
        # for i in range(len(input_sequence)):
        #     for j in range(len(input_sequence)):
        #         line = Line(
        #             token_boxes[i].get_center(),
        #             token_boxes[j].get_center(),
        #             color=ATTENTION_COLOR,
        #             stroke_opacity=0.3,
        #         )
        #         attention_lines.add(line)

        # self.play(Write(transformer_title), *[Create(line) for line in attention_lines])

        # Move the parallel processing visualization below the attention line
        # parallel_boxes = token_boxes.copy()
        # parallel_boxes.next_to(title, DOWN)  # Adjusted position
        # parallel_arrows = VGroup()

        # for box in parallel_boxes:
        #     arrow = Arrow(box.get_top(), box.get_top() + UP, color=TRANSFORMER_COLOR)
        #     parallel_arrows.add(arrow)

        # self.play(
        #     Transform(token_boxes.copy(), parallel_boxes),
        #     *[GrowArrow(arrow) for arrow in parallel_arrows],
        # )

        # # Move advantages and complexity under the parallel boxes
        # advantages = VGroup(
        #     Text("• Parallel processing of all tokens", font_size=20),
        #     Text("• Direct connections between any positions", font_size=20),
        #     Text("• No vanishing gradients", font_size=20),
        #     Text("• Scales better with sequence length", font_size=20),
        # ).arrange(DOWN, aligned_edge=LEFT)
        # advantages.set_color(GREEN)
        # advantages.next_to(parallel_boxes, DOWN)  # Adjusted position

        # self.play(Write(advantages))

        # complexity = VGroup(
        #     MathTex(r"\text{RNN Time Complexity: } O(n)", color=RNN_COLOR),
        #     MathTex(
        #         r"\text{Transformer Time Complexity: } O(1) \text{ with } n \text{ processors}",
        #         color=TRANSFORMER_COLOR,
        #     ),
        # ).arrange(DOWN)
        # complexity.next_to(advantages, DOWN)

        # self.play(Write(complexity))

        comparison = Table(
            [
                ["RNN", "Transformer"],
                ["Sequential", "Parallel"],
                ["Limited Context", "Global Context"],
                ["Memory Efficient", "Computationally Efficient"],
            ],
            include_outer_lines=True,
        ).scale(0.5)
        comparison.next_to(title, DOWN, buff=1.75)

        self.play(Create(comparison))

        # self.play(
        #     attention_lines.animate.set_stroke(opacity=0.8), rate_func=there_and_back
        # )

        final_note = Text(
            "Transformers trade sequential processing for parallel computation",
            color=YELLOW,
            font_size=24,
        )
        final_note.next_to(title, UP, buff=1)  # Changed from DOWN to UP with a buffer

        self.play(Write(final_note))
        self.wait(2)
