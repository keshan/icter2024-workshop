import numpy as np
from manim import *


class LLMTrainingPipeline(Scene):
    def construct(self):

        def create_neural_network(
            num_layers=4, nodes_per_layer=5, scale=0.8, color_gradient=False
        ):
            network = VGroup()
            layers = []

            # Create layers with different colors based on depth
            for i in range(num_layers):
                layer = VGroup()
                for j in range(nodes_per_layer):
                    dot = Dot(radius=0.08)
                    if color_gradient:
                        dot.set_color(
                            color=interpolate_color(
                                BLUE_A, BLUE_E, i / (num_layers - 1)
                            )
                        )
                    layer.add(dot)
                layer.arrange(DOWN, buff=0.3)
                layers.append(layer)

            network.add(*layers)
            network.arrange(RIGHT, buff=1)
            network.scale(scale)

            # Add stylized connections using ArcBetweenPoints
            connections = VGroup()
            for i in range(len(layers) - 1):
                curr_layer = layers[i]
                next_layer = layers[i + 1]
                for curr_node in curr_layer:
                    for next_node in next_layer:
                        line = ArcBetweenPoints(
                            curr_node.get_center(),
                            next_node.get_center(),
                            angle=0.1,
                        ).set_stroke(opacity=0.2)
                        connections.add(line)

            network.add_to_back(connections)
            return network

        def create_data_example(data_type="pretraining"):
            if data_type == "pretraining":
                examples = VGroup(
                    VGroup(
                        Text("Wikipedia:", font_size=16, color=BLUE_B),
                        Text(" Ancient civilizations developed...", font_size=16),
                    ).arrange(RIGHT, buff=0.1),
                    VGroup(
                        Text("Code:", font_size=16, color=GREEN_B),
                        Text(" class NetworkModel(nn.Module):", font_size=16),
                    ).arrange(RIGHT, buff=0.1),
                    VGroup(
                        Text("Book:", font_size=16, color=RED_B),
                        Text(" The character's journey began...", font_size=16),
                    ).arrange(RIGHT, buff=0.1),
                    VGroup(
                        Text("Research:", font_size=16, color=YELLOW_B),
                        Text(" The study demonstrates that...", font_size=16),
                    ).arrange(RIGHT, buff=0.1),
                ).arrange(DOWN, buff=0.3)

                frame = SurroundingRectangle(
                    examples, buff=0.2, color=WHITE, corner_radius=0.2
                )
                return VGroup(examples, frame)

            elif data_type == "finetuning":
                examples = VGroup(
                    VGroup(
                        Text("Clinical:", font_size=16, color=BLUE_B),
                        Text(" Patient presents with...", font_size=16),
                    ).arrange(RIGHT, buff=0.1),
                    VGroup(
                        Text("Diagnosis:", font_size=16, color=GREEN_B),
                        Text(" Tests indicate elevated...", font_size=16),
                    ).arrange(RIGHT, buff=0.1),
                    VGroup(
                        Text("Treatment:", font_size=16, color=RED_B),
                        Text(" Recommended protocol...", font_size=16),
                    ).arrange(RIGHT, buff=0.1),
                ).arrange(DOWN, buff=0.3)

                frame = SurroundingRectangle(
                    examples, buff=0.2, color=WHITE, corner_radius=0.2
                )
                return VGroup(examples, frame)

            else:  # RLHF
                examples = VGroup(
                    VGroup(
                        Text("Human:", font_size=16, color=BLUE_B),
                        Text(" Explain quantum computing", font_size=16),
                    ).arrange(RIGHT, buff=0.1),
                    VGroup(
                        Text("Assistant:", font_size=16, color=GREEN_B),
                        Text(" Quantum computing uses...", font_size=16),
                    ).arrange(RIGHT, buff=0.1),
                    VGroup(
                        Text("Rating:", font_size=16, color=YELLOW_B),
                        Text(" Clear and accurate (9/10)", font_size=16),
                    ).arrange(RIGHT, buff=0.1),
                ).arrange(DOWN, buff=0.3)

                frame = SurroundingRectangle(
                    examples, buff=0.2, color=WHITE, corner_radius=0.2
                )
                return VGroup(examples, frame)

        def create_phase_info(title_text, dataset_size, training_time, compute):
            title = Text(title_text, font_size=36, color=BLUE)
            info = VGroup(
                Text(f"Dataset Size: {dataset_size}", font_size=20),
                Text(f"Training Time: {training_time}", font_size=20),
                Text(f"Compute: {compute}", font_size=20),
            ).arrange(DOWN, buff=0.3, aligned_edge=LEFT)

            return VGroup(title, info).arrange(DOWN, buff=0.5)

        # Initial title with subtle animation
        title = Text("Large Language Model Training Pipeline", font_size=48)
        subtitle = Text(
            "From Raw Data to Intelligent Responses", font_size=28, color=BLUE
        )
        header = VGroup(title, subtitle).arrange(DOWN, buff=0.3)

        self.play(Write(title, run_time=2.5))
        self.play(FadeIn(subtitle, shift=UP), run_time=2)
        self.play(header.animate.scale(0.6).to_edge(UP, buff=0.5), run_time=2)
        self.wait(1.5)

        # Position the network more towards the left initially
        network = create_neural_network(color_gradient=True).shift(LEFT * 2 + DOWN)

        # 1. Pre-training Phase
        pretraining_info = create_phase_info(
            "Pre-training Phase",
            "~1.5 trillion tokens",
            "1-6 months",
            "Thousands of GPUs",
        ).next_to(header, DOWN, buff=0.8)

        # Introduce pre-training phase with animation
        self.play(Write(pretraining_info[0]), run_time=2)  # Slower title write
        self.wait(1)
        self.play(Write(pretraining_info[1]), run_time=3)  # Slower info write
        self.wait(4)  # Extra wait to read

        # Slide info box to the right and move network up
        self.play(
            pretraining_info.animate.shift(RIGHT * 5),
            Create(network),
            run_time=3,  # Slower creation
        )
        self.wait(3)

        # Network moves up slightly
        self.play(network.animate.shift(UP * 2), run_time=1.5)
        self.wait(1)

        # Show data examples with proper timing
        for _ in range(2):
            data_example = create_data_example("pretraining").shift(LEFT * 6 + DOWN)
            self.play(Create(data_example, run_time=2))  # Slower creation
            self.wait(1)  # Wait to read example
            self.play(
                data_example.animate.shift(RIGHT * 2),
                rate_func=smooth,
                run_time=3,  # Slower movement
            )
            self.wait(4)  # Longer wait to process
            self.play(FadeOut(data_example, run_time=1.5))
            self.wait(1)

        # Network learning visualization
        pulses = []
        for layer in network[:-1]:
            pulse = Dot(color=BLUE, radius=0.2)
            pulse.move_to(layer[2])
            pulse.set_opacity(0.8)
            pulses.append(pulse)

        self.play(*[GrowFromCenter(pulse) for pulse in pulses], run_time=2)
        self.play(
            *[pulse.animate.scale(2).set_opacity(0) for pulse in pulses], run_time=2.5
        )
        self.wait(1.5)

        # 2. Fine-tuning Phase
        finetuning_info = create_phase_info(
            "Fine-tuning Phase", "~100k examples", "1-7 days", "8-32 GPUs"
        ).next_to(header, DOWN, buff=0.8)

        # Network moves down as new phase begins
        self.play(
            FadeOut(pretraining_info), network.animate.shift(DOWN * 2), run_time=2
        )
        self.wait(1)

        self.play(Write(finetuning_info[0]), run_time=2)
        self.wait(3)
        self.play(Write(finetuning_info[1]), run_time=3)
        self.wait(4)

        # Slide info box to the right and move network up
        self.play(
            finetuning_info.animate.shift(RIGHT * 5),
            network.animate.shift(UP * 2),
            run_time=2,
        )
        self.wait(1.5)

        # Show specialized medical data examples
        for _ in range(2):
            specialized_data = create_data_example("finetuning").shift(LEFT * 6 + DOWN)
            self.play(Create(specialized_data, run_time=2))
            self.wait(3)  # Wait to read
            self.play(
                specialized_data.animate.shift(RIGHT * 2), rate_func=smooth, run_time=3
            )
            self.wait(4)  # Longer wait to process
            self.play(FadeOut(specialized_data, run_time=1.5))
            self.wait(1)

        # 3. RLHF Phase
        rlhf_info = create_phase_info(
            "RLHF Phase", "~50k human ratings", "2-4 weeks", "16-64 GPUs"
        ).next_to(header, DOWN, buff=0.8)

        # Network moves down as new phase begins
        self.play(FadeOut(finetuning_info), network.animate.shift(DOWN * 2), run_time=2)
        self.wait(1)

        self.play(Write(rlhf_info[0]), run_time=2)
        self.wait(3)
        self.play(Write(rlhf_info[1]), run_time=3)
        self.wait(4)

        # Slide info box to the right and move network up
        self.play(
            rlhf_info.animate.shift(RIGHT * 5),
            network.animate.shift(UP * 2),
            run_time=2,
        )
        self.wait(1.5)

        # Show RLHF examples with ratings
        rlhf_data = create_data_example("rlhf").shift(LEFT * 2 + DOWN * 2)
        self.play(Create(rlhf_data, run_time=2))
        self.wait(4)  # Longer wait to read RLHF example

        # Add reward model
        reward_model = VGroup(
            RoundedRectangle(height=1.5, width=2, corner_radius=0.2),
            Text("Reward\nModel", font_size=20),
        ).arrange(ORIGIN)
        reward_model.next_to(rlhf_data, RIGHT, buff=1)

        # Create feedback system
        feedback_system = VGroup()
        positive = VGroup(
            Circle(radius=0.2, color=GREEN).set_fill(GREEN, opacity=0.2),
            Text("+", font_size=24, color=GREEN),
        ).arrange(ORIGIN, buff=0)

        negative = VGroup(
            Circle(radius=0.2, color=RED).set_fill(RED, opacity=0.2),
            Text("-", font_size=24, color=RED),
        ).arrange(ORIGIN, buff=0)

        feedback = VGroup(positive, negative).arrange(RIGHT, buff=1)

        self.play(Create(feedback.next_to(rlhf_data, DOWN)), run_time=2)
        self.wait(3)
        self.play(Create(reward_model), run_time=2)
        self.wait(3)

        # Animated feedback loop
        arrows = VGroup(
            CurvedArrow(feedback.get_right(), reward_model.get_left(), angle=-0.5),
            CurvedArrow(reward_model.get_top(), network.get_bottom(), angle=-0.5),
        )
        self.play(Create(arrows), run_time=2.5)
        self.wait(4)

        # Final optimization animation
        for _ in range(2):
            self.play(
                *[node.animate.set_color(YELLOW) for node in network[:-1]],
                rate_func=there_and_back,
                run_time=3,  # Slower color transition
            )
            self.wait(1)  # Wait between pulses

        self.wait(4)

        # Elegant fade out
        self.play(
            *[FadeOut(mob, shift=DOWN * 0.5) for mob in self.mobjects], run_time=2.5
        )
        self.wait(2)
