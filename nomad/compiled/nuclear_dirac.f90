!
!
! Evaluates integrals of operators over Dirac test functions
!
!

  !
  ! Returns the overlap of a Dirac delta function and a primitive
  ! Gaussian function
  !
  subroutine overlap(x1, g2, a2, x2, p2, S)
    implicit none
    double precision, intent(in) :: g2
    double precision, intent(in) :: x1(:),a2(:),x2(:),p2(:)
    double complex, intent(out)  :: S

    double precision :: dx(size(x1)), prefactor(size(x1))
    double precision :: real_part(size(x1)), imag_part(size(x1))
    double complex, parameter    :: I    = (0., 1.)
    double precision, parameter  :: pi   = 4 * atan (1.)


    dx        = x1 - x2
    prefactor = (2. * a2 / pi)**(0.25)
    real_part = a2 * dx**2 
    imag_part = p2 * dx
    S = exp(I * g2) * product(prefactor) * exp(sum(-real_part + I*imag_part))

    return
  end subroutine overlap

  !
  ! Returns the del/dp integral of a Dirac delta function and a primitive
  ! Gaussian function
  !
  subroutine deldp(S, x1, x2, int_delp)
    implicit none
    double complex, intent(in)   :: S
    double precision, intent(in) :: x1(:),x2(:)
    double complex, intent(out)  :: int_delp(size(x1))

    double precision :: dx(size(x1))
    double complex, parameter    :: I    = (0., 1.)
 
    dx       = x1 - x2
    int_delp = S * I * dx  

    return
  end subroutine deldp

  !
  ! Returns the del/dx integral of a Dirac delta function and a primitive
  ! Gaussian function
  !
  subroutine deldx(S, x1, a2, x2, p2, int_delx)
    implicit none
    double complex, intent(in)   :: S
    double precision, intent(in) :: x1(:),a2(:),x2(:),p2(:)
    double complex, intent(out)  :: int_delx(size(x1))

    double precision :: dx(size(x1))
    double complex, parameter    :: I    = (0., 1.)
 
    dx       = x1 - x2
    int_delx = S * (-2. * a2 * dx - I * p2) 

    return
  end subroutine deldx

  !
  ! Returns the del^2/d^2x integral of a Dirac delta function and a
  ! primitive Gaussian function
  !
  subroutine deld2x(S, x1, a2, x2, p2, int_del2x)
    implicit none
    double complex, intent(in)   :: S
    double precision, intent(in) :: x1(:),a2(:),x2(:),p2(:)
    double complex, intent(out)  :: int_del2x(size(x1))

    double precision :: dx(size(x1))
    double complex, parameter    :: I    = (0., 1.)
 
    dx        = x1 - x2
    int_del2x = S * (2. * a2 + (2. * a2 * dx + I * p2)**2) 

    return
  end subroutine deld2x
