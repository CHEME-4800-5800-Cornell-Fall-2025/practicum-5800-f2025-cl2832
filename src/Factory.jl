#throw(ErrorException("Oppps! No methods defined in src/Factory.jl. What should you do here?"))
# -- PUBLIC METHODS BELOW HERE ---------------------------------------------------------------------------------------- #
function build(modeltype::Type{MyClassicalHopfieldNetworkModel}, data::NamedTuple)::MyClassicalHopfieldNetworkModel

    # initialize -
    model = modeltype();
    linearimagecollection = data.memories;
    number_of_rows, number_of_cols = size(linearimagecollection);
    W = zeros(Float32, number_of_rows, number_of_rows);
    b = zeros(Float32, number_of_rows); # zero bias for classical Hopfield

    # compute the W -
    for j ∈ 1:number_of_cols # compute the outer product -
        Y = zeros(Float32, number_of_rows, number_of_rows);
        y = linearimagecollection[:,j];
        for i ∈ 1:number_of_rows
            for j ∈ 1:number_of_rows
                if i != j
                Y[i,j] = y[i]*y[j]
                else
                Y[i,j] = 0.0f0; # no self-coupling
                end
            end
        end
        W += Y; # update the W -
    end
    
    WN = (1/number_of_cols)*W; # Hebbian scaling by number of memories stored
    
    # compute the energy dictionary -
    energy = Dict{Int64, Float32}();
    for i ∈ 1:number_of_cols
        energy[i] = _energy(linearimagecollection[:,i], WN, b);
    end

    # add data to the model -
    model.W = WN;
    model.b = b;
    model.energy = energy;

    # return -
    return model;
end
# --- PUBLIC METHODS ABOVE HERE -------------------------------------------------------------------------------------- #