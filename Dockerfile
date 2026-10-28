FROM rust:1.80-bullseye as builder

WORKDIR /usr/src/omni-llm
COPY Cargo.toml Cargo.lock ./
COPY src ./src

RUN cargo build --release

FROM gcr.io/distroless/cc-debian11

COPY --from=builder /usr/src/omni-llm/target/release/omni-llm /usr/local/bin/omni-llm

ENV PORT=8001
EXPOSE 8001

ENTRYPOINT ["omni-llm"]
CMD ["--port", "8001", "--host", "0.0.0.0"]
