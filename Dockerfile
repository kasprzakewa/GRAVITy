FROM julia:1.10.10

WORKDIR /usr/src/julia-service

RUN julia -e 'using Pkg; Pkg.add(["HTTP", "JSON3", "Sockets", "Revise"])'

COPY Project.toml .

RUN julia -e 'using Pkg; Pkg.activate("."); Pkg.instantiate()'

COPY . .

EXPOSE 8001

CMD ["julia", "src/api/server.jl"]