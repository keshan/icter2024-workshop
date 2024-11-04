import numpy as np
from manim import *


class AttentionMechanism(Scene):
    def construct(self):
        # Configuration
        token_color = "#2ecc71"
        query_color = "#e74c3c"
        key_color = "#3498db"
        value_color = "#f1c40f"

        # Create a black background
        bg = Rectangle(
            width=config.frame_width,
            height=config.frame_height,
            fill_color="#000000",  # Changed to pure black
            fill_opacity=1,
            stroke_width=0,
        )
        self.add(bg)

        # Create initial token embedding
        token = Rectangle(height=1.0, width=0.8, color=token_color, fill_opacity=0.3)
        token.move_to(LEFT * 6 + UP * 2)  # Moved higher up

        token_label = Text("Token\nEmbedding", font_size=24, color=token_color)
        token_label.next_to(token, UP)

        # Technical explanation for token embedding
        token_explanation = VGroup(
            Text("Token Embedding:", color="#2ecc71", font_size=24),
            Text(
                "• Dense vector representation of input token",
                font_size=20,
                color="#888888",
            ),
            Text("• Typically 512 or 768 dimensions", font_size=20, color="#888888"),
            Text(
                "• Captures semantic meaning of the token",
                font_size=20,
                color="#888888",
            ),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        token_explanation.to_edge(LEFT).shift(DOWN * 2)  # Move to bottom left

        self.play(
            Create(token), Write(token_label), Write(token_explanation), run_time=1.5
        )
        self.wait(5)  # Give time to read

        # Linear transformations to Q, K, V with arrows
        transformation_explanation = VGroup(
            Text("Linear Transformations:", color="#ffffff", font_size=24),
            Text(
                "• WQ, WK, WV are learned weight matrices",
                font_size=20,
                color="#888888",
            ),
            Text("• Q = TokenEmb × WQ", font_size=20, color="#888888"),
            Text("• K = TokenEmb × WK", font_size=20, color="#888888"),
            Text("• V = TokenEmb × WV", font_size=20, color="#888888"),
            Text(
                "• Typically projects to dk dimensions", font_size=20, color="#888888"
            ),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        transformation_explanation.to_edge(LEFT).shift(DOWN * 2)  # Move to bottom left

        # Create Q, K, V with rectangles
        query = Rectangle(height=1.0, width=0.8, color=query_color, fill_opacity=0.3)
        key = Rectangle(height=1.0, width=0.8, color=key_color, fill_opacity=0.3)
        value = Rectangle(height=1.0, width=0.8, color=value_color, fill_opacity=0.3)

        qkv_group = VGroup(query, key, value).arrange(RIGHT, buff=1.2)
        qkv_group.move_to(LEFT * 2 + UP * 2)

        # Create labels for Q, K, V first
        query_label = Text("Query", color=query_color, font_size=20).next_to(query, UP)
        key_label = Text("Key", color=key_color, font_size=20).next_to(key, UP)
        value_label = Text("Value", color=value_color, font_size=20).next_to(value, UP)

        # Create curved arrows instead of dot paths
        transform_arrows = VGroup(
            Arrow(token.get_right(), query.get_left(), color=query_color),
            Arrow(token.get_right(), key.get_left(), color=key_color),
            Arrow(token.get_right(), value.get_left(), color=value_color),
        )

        # First fade out previous explanation
        self.play(FadeOut(token_explanation), run_time=0.5)

        # Then show new explanation and transformations
        self.play(
            Write(transformation_explanation),
            Create(transform_arrows),
            Transform(token.copy(), query),
            Transform(token.copy(), key),
            Transform(token.copy(), value),
            Write(query_label),
            Write(key_label),
            Write(value_label),
            run_time=2,
        )
        self.wait(6)  # Give time to read

        # Attention calculation explanation
        attention_explanation = VGroup(
            Text("Attention Calculation:", color="#ffffff", font_size=24),
            Text(
                "• Compute compatibility scores between", font_size=20, color="#888888"
            ),
            Text("  Query and all Keys", font_size=20, color="#888888"),
            Text("• Score = (Q × K^T) / √dk", font_size=20, color="#888888"),
            Text(
                "• √dk scaling prevents vanishing gradients",
                font_size=20,
                color="#888888",
            ),
            Text("• Results in attention score matrix", font_size=20, color="#888888"),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        attention_explanation.to_edge(LEFT).shift(DOWN * 2)  # Move to bottom left

        # Attention calculation with matrix visualization
        attention_matrix = Rectangle(
            height=1.0,
            width=1.0,  # Slightly smaller
            fill_color="#9b59b6",
            fill_opacity=0.3,
            stroke_color=WHITE,
        ).move_to(
            LEFT * 3 + UP * 0
        )  # Moved left and kept vertical position

        matrix_grid = VGroup(
            *[
                Line(
                    start=attention_matrix.get_corner(UL) + RIGHT * i / 3 * 1.0,
                    end=attention_matrix.get_corner(DL) + RIGHT * i / 3 * 1.0,
                )
                for i in range(4)
            ]
            + [
                Line(
                    start=attention_matrix.get_corner(UL) + DOWN * i / 3 * 1.0,
                    end=attention_matrix.get_corner(UR) + DOWN * i / 3 * 1.0,
                )
                for i in range(4)
            ]
        )

        attention_label = Text("Attention\nScores", font_size=20).next_to(
            attention_matrix, LEFT
        )

        # Animated attention calculation
        attention_arrows = VGroup(
            CurvedArrow(
                query.get_bottom(), attention_matrix.get_left(), color=query_color
            ),
            CurvedArrow(
                key.get_bottom(), attention_matrix.get_right(), color=key_color
            ),
        )

        # Attention calculation - first fade out previous elements
        self.play(
            FadeOut(transformation_explanation), FadeOut(transform_arrows), run_time=0.5
        )

        # Then show attention calculation
        self.play(
            Write(attention_explanation),
            Create(attention_matrix),
            Create(matrix_grid),
            Write(attention_label),
            Create(attention_arrows),
            run_time=1.5,
        )
        self.wait(6)  # Give time to read

        # Softmax explanation
        softmax_technical = VGroup(
            Text("Softmax Operation:", color="#ffffff", font_size=24),
            Text(
                "• Converts raw scores to probabilities", font_size=20, color="#888888"
            ),
            Text("• exp(xi) / Σexp(xj) for each score", font_size=20, color="#888888"),
            Text(
                "• Output sums to 1.0 (probability dist.)",
                font_size=20,
                color="#888888",
            ),
            Text(
                "• Higher scores get higher probabilities",
                font_size=20,
                color="#888888",
            ),
            Text("• Controls focus on different tokens", font_size=20, color="#888888"),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        softmax_technical.to_edge(LEFT).shift(DOWN * 2)  # Move to bottom left

        # Softmax visualization as probability distribution
        prob_values = [0.45, 0.25, 0.15, 0.10, 0.05]  # Example probability distribution
        max_width = 1.2

        softmax = VGroup(
            *[
                VGroup(
                    Rectangle(
                        height=0.2,
                        width=max_width * prob,
                        fill_color="#3498db",
                        fill_opacity=0.8,
                        stroke_width=1,
                    ),
                    Text(f"{prob:.2f}", font_size=16, color=WHITE).next_to(
                        Rectangle(height=0.2, width=max_width * prob), RIGHT, buff=0.1
                    ),
                ).arrange(RIGHT, buff=0)
                for prob in prob_values
            ]
        ).arrange(DOWN, buff=0.1)

        softmax.next_to(attention_matrix, DOWN, buff=0.5)

        softmax_label = (
            VGroup(
                Text("Softmax Distribution", font_size=20),
                Text(
                    "(Attention weights across tokens)", font_size=16, color="#888888"
                ),
            )
            .arrange(DOWN, buff=0.1)
            .next_to(softmax, LEFT)
        )

        softmax_arrow = Arrow(
            attention_matrix.get_bottom(), softmax.get_top(), color=WHITE
        )

        # Add explanation text
        explanation = Text(
            "→ Normalized attention scores form a\n   probability distribution (Σ = 1.0)",
            font_size=16,
            color="#888888",
        ).next_to(softmax, RIGHT, buff=0.5)

        # Softmax - first fade out previous elements
        self.play(
            FadeOut(attention_explanation), FadeOut(attention_arrows), run_time=0.5
        )

        # Then show softmax explanation and visualization
        self.play(
            Write(softmax_technical),
            Create(softmax),
            Write(softmax_label),
            Create(softmax_arrow),
            # Write(explanation),
            run_time=2,
        )
        self.wait(6)

        # Weighted sum explanation
        output_explanation = VGroup(
            Text("Output Computation:", color="#ffffff", font_size=24),
            Text("• Weighted sum of Values", font_size=20, color="#888888"),
            Text("• Weights from softmax distribution", font_size=20, color="#888888"),
            Text(
                "• Output = Σ(attention_prob_i × value_i)",
                font_size=20,
                color="#888888",
            ),
            Text(
                "• Aggregates information from relevant", font_size=20, color="#888888"
            ),
            Text("  tokens based on attention weights", font_size=20, color="#888888"),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        output_explanation.to_edge(LEFT).shift(DOWN * 2)  # Move to bottom left

        # Final output
        output = Rectangle(height=1.0, width=0.8, color="#1abc9c", fill_opacity=0.3)
        output_glow = output.copy().set_stroke(width=20, opacity=0.3)
        output_group = VGroup(output_glow, output)
        output_group.next_to(softmax, DOWN, buff=0.5)  # Reduced buffer

        output_label = Text("Output", font_size=20, color="#1abc9c").next_to(
            output, RIGHT
        )

        weighted_sum_arrows = VGroup(
            CurvedArrow(softmax.get_bottom(), output.get_left(), color="#9b59b6"),
            CurvedArrow(value.get_bottom(), output.get_right(), color=value_color),
        )

        # Output computation - first fade out previous elements
        self.play(FadeOut(softmax_technical), run_time=0.5)

        # Then show output computation
        self.play(
            Write(output_explanation),
            Create(output_group),
            Write(output_label),
            Create(weighted_sum_arrows),
            run_time=2,
        )
        self.wait(6)

        # Final glow effect with conclusion
        conclusion = VGroup(
            Text("Self-Attention Summary:", color="#ffffff", font_size=24),
            Text(
                "• Dynamically focuses on relevant tokens",
                font_size=20,
                color="#888888",
            ),
            Text("• Captures contextual relationships", font_size=20, color="#888888"),
            Text(
                "• Parallel computation for efficiency", font_size=20, color="#888888"
            ),
            Text("• Foundation of modern transformers", font_size=20, color="#888888"),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        conclusion.to_edge(LEFT).shift(DOWN * 2)  # Move to bottom left

        # Conclusion - first fade out previous elements
        self.play(
            FadeOut(output_explanation), FadeOut(weighted_sum_arrows), run_time=0.5
        )

        # Then show conclusion
        self.play(
            Write(conclusion),
            output_glow.animate.scale(1.2).set_opacity(0.5),
            # rate_func=there_and_back,
            run_time=5,
        )
        self.wait(6)

        # Elegant fade out
        self.play(
            *[FadeOut(mob, shift=DOWN * 0.5) for mob in self.mobjects], run_time=1.5
        )


class MultiHeadAttention(Scene):
    def construct(self):
        # Create black background
        bg = Rectangle(
            width=config.frame_width,
            height=config.frame_height,
            fill_color="#000000",
            fill_opacity=1,
            stroke_width=0,
        )
        self.add(bg)

        # Initial explanation
        initial_explanation = VGroup(
            Text("Multi-Head Attention", color="#ffffff", font_size=32),
            Text(
                "Parallel processing of attention mechanisms",
                font_size=20,
                color="#888888",
            ),
        ).arrange(DOWN, buff=0.5)
        initial_explanation.to_edge(UP, buff=1)

        self.play(Write(initial_explanation))
        self.wait(2)

        # Fade out subheading before showing embeddings
        self.play(FadeOut(initial_explanation[1]))
        self.wait(1)

        # Create input token visualization
        input_token = Rectangle(
            height=1.0, width=0.8, color="#2ecc71", fill_opacity=0.3
        )
        input_token.move_to(LEFT * 6)
        input_label = Text("Input\nToken", font_size=20, color="#2ecc71").next_to(
            input_token, UP
        )

        self.play(Create(input_token), Write(input_label))

        # Create multiple attention heads with better visualization
        heads = 4
        head_colors = ["#FF5733", "#33FF57", "#3357FF", "#FF33F5"]

        # Create mini attention mechanisms for each head
        def create_attention_head(color, index, total):
            y_offset = (total - 1) / 2 - index  # For vertical staggering
            head_group = VGroup()

            # Q, K, V boxes
            qkv = VGroup(
                *[
                    Rectangle(height=0.4, width=0.3, color=color, fill_opacity=0.3)
                    for _ in range(3)
                ]
            ).arrange(RIGHT, buff=0.2)
            qkv.move_to(LEFT * 2 + UP * y_offset)

            # Labels
            labels = VGroup(
                *[
                    Text(t, font_size=14, color=color).next_to(box, UP, buff=0.1)
                    for t, box in zip(["Q", "K", "V"], qkv)
                ]
            )

            # Attention box
            att_box = Rectangle(height=0.6, width=0.6, color=color, fill_opacity=0.2)
            att_box.next_to(qkv, RIGHT, buff=1)
            att_label = Text(f"Head {index+1}", font_size=16, color=color).next_to(
                att_box, UP
            )

            # Arrows
            input_arrows = VGroup(
                *[
                    Arrow(
                        input_token.get_right(),
                        box.get_left(),
                        color=color,
                        stroke_width=2,
                    )
                    for box in qkv
                ]
            )

            att_arrow = Arrow(
                qkv[0].get_right(), att_box.get_left(), color=color, stroke_width=2
            )

            head_group.add(qkv, labels, att_box, att_label, input_arrows, att_arrow)
            return head_group, att_box

        # Create and position all heads
        attention_heads = VGroup()
        output_boxes = []
        for i in range(heads):
            head_group, out_box = create_attention_head(head_colors[i], i, heads)
            attention_heads.add(head_group)
            output_boxes.append(out_box)

        # Show parallel processing with staggered animation
        for head in attention_heads:
            self.play(Create(head), run_time=0.75)
        self.wait(2)

        # Technical explanation
        head_explanation = VGroup(
            Text("Parallel Attention Heads:", color="#ffffff", font_size=24),
            Text(
                "• Each head learns different patterns", font_size=20, color="#888888"
            ),
            Text("• Different aspects of relationships", font_size=20, color="#888888"),
            Text("• Independent parameter matrices", font_size=20, color="#888888"),
        ).arrange(DOWN, aligned_edge=RIGHT)
        head_explanation.to_edge(RIGHT).shift(DOWN)

        self.play(Write(head_explanation))
        self.wait(7)

        # Concatenation visualization
        concat = Rectangle(height=2, width=1.2, color="#e67e22", fill_opacity=0.3)
        concat.next_to(
            output_boxes[1], RIGHT, buff=1.5
        )  # Position relative to middle heads
        concat_label = Text("Concatenate", font_size=20, color="#e67e22").next_to(
            concat, UP
        )

        # Animated concatenation arrows
        concat_arrows = VGroup(
            *[
                Arrow(box.get_right(), concat.get_left(), color=head_colors[i])
                for i, box in enumerate(output_boxes)
            ]
        )

        self.play(
            FadeOut(head_explanation),
            Create(concat),
            Write(concat_label),
            *[GrowArrow(arrow) for arrow in concat_arrows],
            run_time=1.5,
        )

        # Concatenation explanation
        concat_explanation = VGroup(
            Text("Concatenation:", color="#ffffff", font_size=24),
            Text("• Combine all head outputs", font_size=20, color="#888888"),
            Text(
                "• Preserve information from all heads", font_size=20, color="#888888"
            ),
            Text("• [Head₁; Head₂; Head₃; Head₄]", font_size=20, color="#888888"),
        ).arrange(DOWN, aligned_edge=RIGHT)
        concat_explanation.to_edge(RIGHT).shift(DOWN)

        self.play(Write(concat_explanation))
        self.wait(7)

        # Final projection
        final_output = Rectangle(
            height=1.5, width=1.2, color="#16a085", fill_opacity=0.3
        )
        final_output.next_to(concat, RIGHT, buff=1.5)
        final_label = Text("Linear\nProjection", font_size=20, color="#16a085").next_to(
            final_output, UP
        )

        # Add glow effect to final output
        final_glow = final_output.copy()
        final_glow.set_stroke(color="#16a085", opacity=0.3, width=20)
        final_group = VGroup(final_glow, final_output)

        final_arrow = Arrow(
            concat.get_right(), final_output.get_left(), color="#e67e22"
        )

        self.play(
            FadeOut(concat_explanation),
            Create(final_group),
            Write(final_label),
            Create(final_arrow),
        )

        # Final explanation
        final_explanation = VGroup(
            Text("Final Projection:", color="#ffffff", font_size=24),
            Text("• Projects concatenated features", font_size=20, color="#888888"),
            Text("• Combines multi-head information", font_size=20, color="#888888"),
            Text("• Produces final attention output", font_size=20, color="#888888"),
        ).arrange(DOWN, aligned_edge=RIGHT)
        final_explanation.to_edge(RIGHT).shift(DOWN)

        self.play(Write(final_explanation))
        self.wait(7)

        # Final glow effect
        self.play(
            final_glow.animate.scale(1.2).set_opacity(0.5),
            rate_func=there_and_back,
            run_time=2,
        )
        self.wait(2)

        # Elegant fade out
        self.play(
            *[FadeOut(mob, shift=DOWN * 0.5) for mob in self.mobjects], run_time=1.5
        )
