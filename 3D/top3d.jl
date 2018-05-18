using PyPlot
using PyCall
using CholmodSolve2


include("ndgrid.jl")
# AN 169 LINE 3D TOPOLOGY OPITMIZATION CODE BY LIU AND TOVAR [JUL 2013]
function top3d(nelx::Int,nely::Int,nelz::Int,volfrac::Real,penal::Real,rmin::Real)
  # USER-DEFINED LOOP PARAMETERS
  maxloop = 200;    # Maximum number of iterations
  tolx = 0.01;      # Terminarion criterion
  displayflag = false;  # Display structure flag
  # USER-DEFINED MATERIAL PROPERTIES
  E0 = 1;           # Young's modulus of solid material()
  Emin = 1e-9;      # Young's modulus of void-like material()
  nu = 0.3;         # Poisson's ratio
  # USER-DEFINED LOAD DOFs
  il,jl,kl = meshgrid([nelx], [0], 0:nelz);                 # Coordinates
  # il = [30,30,30]  #fill(nelx,nelz+1)
  # jl = [0,0,0]  #fill(0,nelz+1)
  # kl = 0:nelz
  loadnid = kl*(nelx+1)*(nely+1)+il*(nely+1)+(nely+1-jl); # Node IDs
  loaddof = 3*loadnid[:] - 1;                             # DOFs
  # USER-DEFINED SUPPORT FIXED DOFs
  iif,jf,kf = meshgrid([0],0:nely,0:nelz);                  # Coordinates
  fixednid = kf*(nelx+1)*(nely+1)+iif*(nely+1)+(nely+1-jf); # Node IDs
  fixeddof = [3*fixednid[:]; 3*fixednid[:]-1; 3*fixednid[:]-2]; # DOFs
  # PREPARE FINITE ELEMENT ANALYSIS
  nele = nelx*nely*nelz
  ndof = 3*(nelx+1)*(nely+1)*(nelz+1)
  # F = sparse(loaddof,1.0*ones(length(loaddof)),-1,ndof,1,)
  F = zeros(Float64,ndof)
  F[loaddof] = -1.0
  U = zeros(Float64,ndof)
  freedofs = setdiff(1:ndof,fixeddof)
  KE = lk_H8(nu)
  nodegrd = reshape(1:(nely+1)*(nelx+1),nely+1,nelx+1)
  nodeids = reshape(nodegrd[1:end-1,1:end-1],nely*nelx,1)
  nodeidz = 0:(nely+1)*(nelx+1):(nelz-1)*(nely+1)*(nelx+1)
  nodeidz = reshape(nodeidz, 1,length(nodeidz))
  nodeids = repmat(nodeids,size(nodeidz)[1],size(nodeidz)[2])+repmat(nodeidz,size(nodeids)[1],size(nodeids)[2])
  edofVec = 3*nodeids[:]+1
  edofMat = repmat(edofVec,1,24)+ repmat([0 1 2 3*nely + [3 4 5 0 1 2] -3 -2 -1 3*(nely+1)*(nelx+1)+[0 1 2 3*nely + [3 4 5 0 1 2] -3 -2 -1]],nele,1)
  iK = reshape(kron(edofMat,ones(24,1))',24*24*nele)
  jK = reshape(kron(edofMat,ones(1,24))',24*24*nele)
  # PREPARE FILTER
  # xxx = nele*(2*(ceil(rmin)-1)+1)^2
  iH = Array{Float64,1}(convert(Int64,nele*(2*(ceil(rmin)-1)+1)^2))
  jH = Array{Float64,1}(length(iH))
  sH = Array{Float64,1}(length(iH))
  # iH = 1.0*ones(nele*(2*(ceil(rmin)-1)+1)^2)
  # jH = 1.0*ones(size(iH))
  # sH = 1.0*zeros(size(iH))
  k = 0
   @simd for k1 = 1:nelz
     @simd for i1 = 1:nelx
       @simd for j1 = 1:nely
        e1 = (k1-1)*nelx*nely + (i1-1)*nely+j1
        @inbounds @simd for k2 = max(k1-(ceil(rmin)-1),1):min(k1+(ceil(rmin)-1),nelz)
          @inbounds @simd for i2 = max(i1-(ceil(rmin)-1),1):min(i1+(ceil(rmin)-1),nelx)
            @inbounds @simd for j2 = max(j1-(ceil(rmin)-1),1):min(j1+(ceil(rmin)-1),nely)
              @fastmath e2 = (k2-1)*nelx*nely + (i2-1)*nely+j2
              k = k+1
              if k<=length(iH)
              @inbounds  iH[k] = e1
              @inbounds  jH[k] = e2
              @inbounds  @fastmath sH[k] = max(0,rmin-sqrt((i1-i2)^2+(j1-j2)^2+(k1-k2)^2))
              else
              @inbounds  push!(iH,e1)
              @inbounds  push!(jH,e2)
              @inbounds  @fastmath push!(sH,max(0,rmin-sqrt((i1-i2)^2+(j1-j2)^2+(k1-k2)^2)))
              end
            end
          end
        end
      end
    end
  end
  H = sparse(iH,jH,sH)
  Hs = sparse(sum(H,2))
  # INITIALIZE ITERATION
  @inbounds  x = repeat([volfrac],outer=[nely,nelx,nelz])
  xPhys = copy(x);
  loop = 0;
  change = 1
  # START ITERATION
  while change > tolx && loop < maxloop
    loop = loop+1
    # FE-ANALYSIS
    @inbounds sK = reshape(KE[:]*(Emin+xPhys[:]'.^penal*(E0-Emin)),24*24*nele)

    @inbounds K = sparse(iK,jK,sK)
    @fastmath K = (K+K')/2

    @fastmath K_fac = ldltfact(K[freedofs,freedofs])
    @fastmath U[freedofs] = A_ldiv_B!(U[freedofs],K_fac, F[freedofs])

    # U[freedofs,:] = sparse(pinv(a)*b)
    # OBJECTIVE FUNCTION AND SENSITIVITY ANALYSIS
    @inbounds @fastmath ce = reshape(sum((U[edofMat]*KE).*U[edofMat],2),nely,nelx,nelz)
    @inbounds @fastmath c = sum(sum(sum((Emin+xPhys.^penal*(E0-Emin)).*ce)))
    @inbounds @fastmath dc = -penal*(E0-Emin)*xPhys.^(penal-1).*ce
    dv = ones(nely,nelx,nelz)
    # FILTERING AND MODIFICATION OF SENSITIVITIES
    @inbounds @fastmath dc[:] = H*(dc[:]./Hs);
    @inbounds @fastmath dv[:] = H*(dv[:]./Hs)
    # OPTIMALITY CRITERIA UPDATE
    l1 = 0; l2 = 1e9; move = 0.2
    xnew = Float64[]
    while (l2-l1)/(l1+l2) > 1e-3
      @fastmath lmid = 0.5*(l2+l1)
      @inbounds @fastmath xnew = max.(0,max.(x-move,min.(1,min.(x+move,x.*sqrt.(-dc./dv/lmid)))))
      @inbounds @fastmath xPhys[:] = (H*xnew[:])./Hs
      if sum(xPhys) > volfrac*nele
        l1 = lmid;
      else
        l2 = lmid;
      end
    end
    @fastmath change = maximum(abs.(xnew[:]-x[:]))
    x = xnew
    # PRINT RESULTS
    # printf(" It.:%5i Obj.:%11.4f Vol.:%7.3f ch.:%7.3f\n",loop,c,mean(xPhys[:]),change)
    println(" It.:$loop Obj.:$c Vol.:$(mean(xPhys[:])) ch.:$change\n")
    # PLOT DENSITIES
    if displayflag
      display_3D(xPhys)
    end ##ok<UNRCH>
  end
    display(xPhys)
    display_3D(xPhys)
end


  # === GENERATE ELEMENT STIFFNESS MATRIX ===
function lk_H8(nu)
  A = [32 6 -8 6 -6 4 3 -6 -10 3 -3 -3 -4 -8;
      -48 0 0 -24 24 0 0 0 12 -12 0 12 12 12]
  k = (1/144)*A'*[1; nu]

  K1 = [k[1] k[2] k[2] k[3] k[5] k[5];
      k[2] k[1] k[2] k[4] k[6] k[7];
      k[2] k[2] k[1] k[4] k[7] k[6];
      k[3] k[4] k[4] k[1] k[8] k[8];
      k[5] k[6] k[7] k[8] k[1] k[2];
      k[5] k[7] k[6] k[8] k[2] k[1]]
  K2 = [k[9]  k[8]  k[12] k[6]  k[4]  k[7];
      k[8]  k[9]  k[12] k[5]  k[3]  k[5];
      k[10] k[10] k[13] k[7]  k[4]  k[6];
      k[6]  k[5]  k[11] k[9]  k[2]  k[10];
      k[4]  k[3]  k[5]  k[2]  k[9]  k[12];
      k[11] k[4]  k[6]  k[12] k[10] k[13]]
  K3 = [k[6]  k[7]  k[4]  k[9]  k[12] k[8];
      k[7]  k[6]  k[4]  k[10] k[13] k[10];
      k[5]  k[5]  k[3]  k[8]  k[12] k[9];
      k[9]  k[10] k[2]  k[6]  k[11] k[5];
      k[12] k[13] k[10] k[11] k[6]  k[4];
      k[2]  k[12] k[9]  k[4]  k[5]  k[3]]
  K4 = [k[14] k[11] k[11] k[13] k[10] k[10];
      k[11] k[14] k[11] k[12] k[9]  k[8];
      k[11] k[11] k[14] k[12] k[8]  k[9];
      k[13] k[12] k[12] k[14] k[7]  k[7];
      k[10] k[9]  k[8]  k[7]  k[14] k[11];
      k[10] k[8]  k[9]  k[7]  k[11] k[14]]
  K5 = [k[1] k[2]  k[8]  k[3] k[5]  k[4];
      k[2] k[1]  k[8]  k[4] k[6]  k[11];
      k[8] k[8]  k[1]  k[5] k[11] k[6];
      k[3] k[4]  k[5]  k[1] k[8]  k[2];
      k[5] k[6]  k[11] k[8] k[1]  k[8];
      k[4] k[11] k[6]  k[2] k[8]  k[1]]
  K6 = [k[14] k[11] k[7]  k[13] k[10] k[12];
      k[11] k[14] k[7]  k[12] k[9]  k[2];
      k[7]  k[7]  k[14] k[10] k[2]  k[9];
      k[13] k[12] k[10] k[14] k[7]  k[11];
      k[10] k[9]  k[2]  k[7]  k[14] k[7];
      k[12] k[2]  k[9]  k[11] k[7]  k[14]]
      # careful with this next line of code
      # see differences at https://docs.julialang.org/en/stable/manual/noteworthy-differences/#Noteworthy-differences-from-MATLAB-1
  @fastmath KE = 1/((nu+1)*(1-2*nu))*
  [ K1  K2  K3  K4;
      K2'  K5  K6  K3';
      K3' K6  K5' K2';
      K4  K3  K2  K1';]
      return KE
end

function display_3D(rho)
  nely,nelx,nelz = size(rho)
  # display(rho)
  hx = 1; hy = 1; hz = 1;            # User-defined unit element size()
  # face = [1 2 3 4; 2 6 7 3; 4 3 7 8; 1 5 8 4; 1 2 6 5; 5 6 7 8]
  # set(gcf(),"Name','ISO display()','NumberTitle','off")
  PyObject(PyPlot.axes3D)
  cfig = figure()
  ax = cfig[:add_subplot](111, projection="3d")
  for k = 1:nelz
      z = (k-1)*hz
      for i = 1:nelx
          x = (i-1)*hx
          for j = 1:nely
              y = nely*hy - (j-1)*hy
              if rho[j,i,k] > 0.5  # User-defined display density threshold
                  vert = [x y z; x y-hx z; x+hx y-hx z; x+hx y z; x y z+hx;x y-hx z+hx; x+hx y-hx z+hx;x+hx y z+hx]
                  vert[:,[2 3]] = vert[:,[3 2]]; vert[:,2,:] = -vert[:,2,:]
                  # patch("Faces',face,'Vertices',vert,'FaceColor",[0.2+0.8*(1-rho[j,i,k]),0.2+0.8*(1-rho[j,i,k]),0.2+0.8*(1-rho[j,i,k])])
                  # hold on
                  # display(vert)

                  # ax[:scatter3D](vert[:, 1], vert[:, 2], vert[:, 3])
                  verts = [
                   [vert[1,:] vert[2,:] vert[3,:] vert[4,:]]',
                   [vert[2,:] vert[6,:] vert[7,:] vert[3,:]]',
                   [vert[4,:] vert[3,:] vert[7,:] vert[8,:]]',
                   [vert[1,:] vert[5,:] vert[8,:] vert[4,:]]',
                   [vert[1,:] vert[2,:] vert[6,:] vert[5,:]]',
                   [vert[5,:] vert[6,:] vert[7,:] vert[8,:]]'
                   ]
                  ax[:add_collection3d](art3D[:Poly3DCollection](verts, facecolor=[0.2+0.8*(1-rho[j,i,k]),0.2+0.8*(1-rho[j,i,k]),0.2+0.8*(1-rho[j,i,k])], linewidth=1, edgecolor="black"))

              end
          end
      end
  end
  #axis equal; axis tight; axis off; box on; view([30,30]); pause(1e-6)
  ax[:autoscale]()
  ax[:axis]("equal")
  ax[:axis]("tight")
  ax[:view_init](elev=30, azim=30)
  ax[:dist]=200


end

  # =========================================================================
  # === This code is translated from the matlab code written by K Liu and
  # === A Tovar, Dept. of Mechanical Engineering, IUPUI,   ===
  # === The code is intended for educational purposes, and the details    ===
  # === and extensions can be found in the paper:                         ===
  # === K. Liu and A. Tovar, "An efficient 3D topology optimization code  ===
  # === written in Matlab", Struct Multidisc Optim, 50[6]: 1175-1196, 2014, =
  # === doi:10.1007/s00158-014-1107-x                                     ===
  # === ----------------------------------------------------------------- ===
  # === The code as well as an uncorrected version of the paper can be    ===
  # === downloaded from the website: http://www.top3dapp.com/             ===
  # === ----------------------------------------------------------------- ===
  # === Disclaimer:                                                       ===
  # === The authors reserves all rights for the program.                  ===
  # === The code may be distributed and used for educational purposes.    ===
  # === The authors do not guarantee that the code is free from errors, a
# @enter top3d(30,10,2,0.5,3.0,1.2)
# top3d(30,10,2,0.5,3.0,1.2)
# @profile top3d(30,10,2,0.5,3.0,1.2)
# @time top3d(30,10,2,0.5,3.0,1.2)
# @enter top3d(60, 20, 4, 0.3, 3, 1.5)
# top3d(60, 20, 4, 0.3, 3, 1.5)
top3d(40,20,5,0.30,3.0,1.5)
