import numpy as np
from manim import *


class EnhancedWordEmbeddingAnimation(Scene):
    def create_embedding_vector(self, values, color, height=0.4, width=0.8):
        vector = VGroup()
        for i, val in enumerate(values):
            cell = Rectangle(height=height, width=width, stroke_width=1)
            text = DecimalNumber(val, num_decimal_places=2, font_size=16)
            text.move_to(cell)
            group = VGroup(cell, text)
            group.set_color(color)
            vector.add(group)
        vector.arrange(DOWN, buff=0.1)
        return vector

    def create_annotation(self, text, color, font_size=20):
        return Text(text, color=color, font_size=font_size)

    def construct(self):
        # Define colors
        WORD_COLOR = "#2ecc71"
        TOKEN_COLOR = "#e74c3c"
        EMBED_COLOR = "#3498db"
        POS_COLOR = "#9b59b6"
        FINAL_COLOR = "#f1c40f"
        DIM_COLOR = "#95a5a6"

        # Create title and subtitle
        title = Text("The journey of a Token", font_size=40).to_edge(UP, buff=0.5)
        subtitle = Text(
            "From Words to High-Dimensional Vectors", font_size=24, color=DIM_COLOR
        )
        subtitle.next_to(title, DOWN)

        self.play(Write(title), Write(subtitle))

        # Input words section
        input_words = ["Hello", "ICTer", "Workshop"]
        word_groups = VGroup()
        for word in input_words:
            text = Text(word, font_size=32, color=WORD_COLOR)
            box = SurroundingRectangle(text, color=WORD_COLOR, corner_radius=0.2)
            group = VGroup(text, box)
            word_groups.add(group)

        word_groups.arrange(DOWN, buff=0.5)
        word_groups.to_edge(LEFT, buff=1)

        # Mathematical notation for embedding
        embed_formula = MathTex(
            r"E: V \rightarrow \mathbb{R}^d", r"\quad d = 512", color=EMBED_COLOR
        ).scale(0.8)
        embed_formula.next_to(subtitle, DOWN, buff=1)

        # Add explanatory text
        explanation = Text(
            "Each word is mapped to a high-dimensional vector",
            font_size=20,
            color=DIM_COLOR,
        ).next_to(embed_formula, DOWN)

        self.play(FadeIn(word_groups), Write(embed_formula), Write(explanation))

        # Wait for 2 seconds
        self.wait(3)

        # Remove embed_formula and explanation
        self.play(FadeOut(embed_formula), FadeOut(explanation))
        # Process first word as example
        first_word = word_groups[0]
        # Tokenization with mathematical notation
        token_process = MathTex(
            r"\text{tokenize}(", r"\text{Hello}", r") = ", r"35674", color=TOKEN_COLOR
        ).scale(0.8)
        token_process.next_to(first_word, RIGHT, buff=2)

        self.play(
            Write(token_process), first_word.animate.set_color(YELLOW), run_time=1.25
        )
        self.play(first_word.animate.set_color(WORD_COLOR))

        # Create embedding vectors (showing more dimensions)
        embedding_values = np.random.randn(8) * 0.5
        pos_values = np.sin(np.linspace(0, 1, 8)) * 0.3
        final_values = embedding_values + pos_values

        # Create vectors with labels
        embedding_vector = self.create_embedding_vector(embedding_values, EMBED_COLOR)
        pos_vector = self.create_embedding_vector(pos_values, POS_COLOR)
        final_vector = self.create_embedding_vector(final_values, FINAL_COLOR)

        # Arrange vectors
        vectors_group = VGroup(embedding_vector, pos_vector, final_vector)
        vectors_group.arrange(RIGHT, buff=1.5)
        vectors_group.next_to(token_process, DOWN, buff=1.5)

        # Labels for vectors
        embed_label = Text("Token\nEmbedding", color=EMBED_COLOR, font_size=20)
        pos_label = Text("Position\nEncoding", color=POS_COLOR, font_size=20)
        final_label = Text("Final\nEmbedding", color=FINAL_COLOR, font_size=20)

        labels = VGroup(embed_label, pos_label, final_label)
        for label, vector in zip(labels, vectors_group):
            label.next_to(vector, UP)

        # Positional encoding formula
        pos_formula = MathTex(
            r"PE_{(pos,2i)} &= \sin\left(\frac{pos}{10000^{2i/d}}\right)",
            r"\\PE_{(pos,2i+1)} &= \cos\left(\frac{pos}{10000^{2i/d}}\right)",
            color=POS_COLOR,
        ).scale(0.6)
        pos_formula.next_to(vectors_group, DOWN, buff=1)

        # Show vectors and formula
        self.play(
            *[Create(vector) for vector in vectors_group],
            *[Write(label) for label in labels],
            Write(pos_formula),
        )

        # Dimension indicators
        dim_arrows = VGroup()
        dim_labels = VGroup()
        for i, vector in enumerate(vectors_group):
            arrow = DoubleArrow(
                vector.get_top() + UP * 0.2,
                vector.get_bottom() + DOWN * 0.2,
                color=DIM_COLOR,
                buff=0.1,
            ).scale(0.5)
            arrow.next_to(vector, RIGHT, buff=0.1)
            label = Text(f"d={len(embedding_values)}", font_size=16, color=DIM_COLOR)
            label.next_to(arrow, RIGHT, buff=0.1)
            dim_arrows.add(arrow)
            dim_labels.add(label)

        self.play(
            *[GrowArrow(arrow) for arrow in dim_arrows],
            *[Write(label) for label in dim_labels],
        )

        # Addition symbols
        plus = Text("+", font_size=36).move_to(pos_vector.get_center() + LEFT * 0.5)
        equals = Text("=", font_size=36).move_to(final_vector.get_center() + LEFT * 0.5)

        self.play(Write(plus), Write(equals))

        # Highlight addition process
        for i in range(len(embedding_values)):
            self.play(
                embedding_vector[i].animate.set_color(YELLOW),
                pos_vector[i].animate.set_color(YELLOW),
                final_vector[i].animate.set_color(YELLOW),
                run_time=0.3,
            )
            self.play(
                embedding_vector[i].animate.set_color(EMBED_COLOR),
                pos_vector[i].animate.set_color(POS_COLOR),
                final_vector[i].animate.set_color(FINAL_COLOR),
                run_time=0.3,
            )

        # Final summary box
        summary_text = """
        Key Points:
        1. Words → Tokens → Embeddings
        2. Embeddings capture semantic meaning
        3. Position adds sequential information
        4. Final dim 
        """
        summary = Text(summary_text, font_size=24, color=WHITE)
        summary_box = SurroundingRectangle(summary, buff=0.3, color=DIM_COLOR)
        summary_group = VGroup(summary, summary_box)
        summary_group.next_to(title, DOWN, buff=1.5)

        # Hide token process, embeddings, and others except the title
        self.play(
            FadeOut(word_groups),
            # FadeOut(embed_formula),
            # FadeOut(explanation),
            FadeOut(token_process),
            FadeOut(vectors_group),
            FadeOut(labels),
            FadeOut(pos_formula),
            FadeOut(dim_arrows),
            FadeOut(dim_labels),
            FadeOut(plus),
            FadeOut(equals),
        )

        self.play(Write(summary), Create(summary_box))

        # Show example of semantic relationships
        semantic_example = MathTex(
            r"\vec{king} - \vec{man} + \vec{woman} \approx \vec{queen}",
            color=EMBED_COLOR,
        ).scale(0.8)
        semantic_example.next_to(summary_group, DOWN)

        self.play(Write(semantic_example))

        # Final emphasis on the result
        # self.play(final_vector.animate.scale(1.1), rate_func=there_and_back)

        self.wait(3)
