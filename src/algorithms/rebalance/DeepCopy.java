package algorithms.rebalance;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;

/*
 * JBoss, Home of Professional Open Source
 * Copyright 2005, JBoss Inc., and individual contributors as indicated
 * by the @authors tag. See the copyright.txt in the distribution for a
 * full listing of individual contributors.
 *
 * This is free software; you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation; either version 2.1 of
 * the License, or (at your option) any later version.
 *
 * This software is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this software; if not, write to the Free
 * Software Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA
 * 02110-1301 USA, or see the FSF site: http://www.fsf.org.
 */

public class DeepCopy {

  // ///////////////////////////////////////////////////////////////////////
  // Cloning Methods //
  // ///////////////////////////////////////////////////////////////////////

  /**
   * Copy an serializable object deeply.
   * 
   * @param obj
   *          Object to copy.
   * @return Copied object.
   * 
   * @throws IOException
   * @throws ClassNotFoundException
   */
  public static Object copy(final Serializable obj) throws IOException, ClassNotFoundException {
    ObjectOutputStream out = null;
    ObjectInputStream in = null;
    Object copy = null;

    try {
      // write the object
      ByteArrayOutputStream baos = new ByteArrayOutputStream();
      out = new ObjectOutputStream(baos);
      out.writeObject(obj);
      out.flush();

      // read in the copy
      byte data[] = baos.toByteArray();
      ByteArrayInputStream bais = new ByteArrayInputStream(data);
      in = new ObjectInputStream(bais);
      copy = in.readObject();
    } finally {
      out.close();
      in.close();
    }

    return copy;
  }
}